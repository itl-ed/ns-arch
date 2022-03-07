"""
Implement custom modules by extending detectron2-provided defaults, to be plugged
into our scene graph generation model
"""
import torch
import numpy as np
from detectron2.layers import ShapeSpec
from detectron2.config import configurable
from detectron2.layers import cat
from detectron2.structures import pairwise_iou, Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    build_box_head
)

from .fast_rcnn import SceneGraphRCNNOutputLayers
from ..utils import pair_vals


@ROI_HEADS_REGISTRY.register()
class SceneGraphROIHeads(StandardROIHeads):
    """
    ROIHeads extended to produce outputs (inference results & training losses)
    for the scene graph generation task; i.e. in addition to box regression and
    class prediction, adds attribute & relation prediction

    Re-implement:
        __init__(), from_config(), _init_box_head(), _sample_proposals(),
        label_and_sample_proposals(), _forward_box(), forward()
    """
    @configurable
    def __init__(
        self,
        *,
        num_attributes,
        num_relations,
        **kwargs
    ):
        """
        Add info about numbers of attribute/relation predicates as well
        """
        super().__init__(**kwargs)
        self.num_attributes = num_attributes
        self.num_relations = num_relations

    @classmethod
    def from_config(cls, cfg, input_shape):
        """
        Add info about numbers of attribute/relation predicates as well
        """
        ret = super().from_config(cfg, input_shape)
        ret["num_attributes"] = cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES
        ret["num_relations"] = cfg.MODEL.ROI_HEADS.NUM_RELATIONS
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        """
        Minimally extended to replace the hard-coded FastRCNNOutputLayers module to our custom
        SceneGraphRCNNOutputLayers module; otherwise identical
        """
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = SceneGraphRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes, gt_attributes):
        """
        Extended to handle multi-label predictions, and return ground-truth attributes
        """
        has_gt_cats = len(gt_classes) > 0
        # Get the corresponding GT for each proposal
        if has_gt_cats:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0, :] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1, :] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
            gt_classes = gt_classes.unsqueeze(-1).expand(-1, self.num_classes)  # Match dims

        # Do the same for attributes as well
        has_gt_atts = len(gt_attributes) > 0
        if has_gt_atts:
            gt_attributes = gt_attributes[matched_idxs]
            # ... but proposals with matched_labels==[0, -1] will be ignored later, since
            # attributes are not defined for non-object proposals
            gt_attributes[matched_labels == 0, :] = self.num_attributes
            gt_attributes[matched_labels == -1, :] = self.num_attributes
        else:
            gt_attributes = torch.zeros_like(matched_idxs) + self.num_attributes
            gt_attributes = gt_attributes.unsqueeze(-1).expand(-1, self.num_attributes)

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes[:,0], self.batch_size_per_image, self.positive_fraction, self.num_classes
        )   # Add [:,0] to the first arg (gt_classes) to convert into expected format

        sampled_idxs = cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs], gt_attributes[sampled_idxs]
    
    def _sample_proposal_pairs(self, matched_idxs, matched_labels, sampled_idxs, gt_relations):
        """
        Helper method to sample pairs of proposals that are matched to foreground classes, from two
        partitions:
            1) pairs that have at least one positive relation annotations
            2) complement of 1) (w.r.t universe of all possible foreground proposal pairs)...

        ... because there are too many (i.e. quadratically many) index pairs
        """
        N_GT = len(gt_relations)   # No. of ground-truth objects

        if N_GT == 0:
            # No ground-truth relations to match
            return torch.empty((0, 2), dtype=torch.int64, device=gt_relations.device)

        # Mapping from proposal sample indices to matched ground-truth object labels & whether
        # samples are matched to foreground (1) or background (0) labels
        # (-1 labels are already ignored and shouldn't exist)
        sampled_targets = matched_idxs[sampled_idxs]
        sampled_labels = matched_labels[sampled_idxs]

        # All possible pairs of foreground proposal indices
        fg_pairs_all = pair_vals((sampled_labels==1).nonzero()).view(-1, 2)

        # Such pairs mapped to corresponding ground-truth objects
        gt_rel_inds = torch.gather(sampled_targets, 0, fg_pairs_all.view(-1)).view(-1, 2)

        # Use the mapped pair indices to filter pairs that have positive relation annotations
        gt_rel_inds = gt_rel_inds[:,0] * N_GT + gt_rel_inds[:,1]
        fg_pairs_pos = torch.gather(gt_relations.sum(-1).view(-1), 0, gt_rel_inds)
        fg_pairs_pos = fg_pairs_all[fg_pairs_pos>0]

        # Set complement to find 'negative' pairs that do not have relation annotations
        pairs_all_set = {(int(pair[0]), int(pair[1])) for pair in fg_pairs_all}
        pairs_with_rels_set = {(int(pair[0]), int(pair[1])) for pair in fg_pairs_pos}
        pairs_without_rels_set = pairs_all_set - pairs_with_rels_set
        fg_pairs_neg = torch.tensor([pair for pair in pairs_without_rels_set], dtype=torch.int64)

        fg_pairs_pos = fg_pairs_pos.to(gt_relations.device)
        fg_pairs_neg = fg_pairs_neg.to(gt_relations.device)

        # Now subsample from the positive set and the negative set, using logic similar to the 
        # subsample_labels() method
        rels_per_image = int(self.batch_size_per_image / 2)
        num_pos = int(rels_per_image * self.positive_fraction)
        num_pos = min(len(fg_pairs_pos), num_pos)
        num_neg = rels_per_image - num_pos
        num_neg = min(len(fg_pairs_neg), num_neg)

        perm1 = torch.randperm(len(fg_pairs_pos), device=fg_pairs_pos.device)[:num_pos]
        perm2 = torch.randperm(len(fg_pairs_neg), device=fg_pairs_neg.device)[:num_neg]
        pos_idx_pairs = fg_pairs_pos[perm1]; neg_idx_pairs = fg_pairs_neg[perm2]

        return cat([pos_idx_pairs, neg_idx_pairs])
    
    def _fetch_rels_for_sampled_pairs(self, matched_idxs, sampled_idxs, sampled_idx_pairs, gt_relations):
        """
        Helper method to map ground-truth relation annotations to the space of sampled proposal pairs
        """
        N_R = self.num_relations
        N_S = len(sampled_idxs)    # No. of proposal samples considered
        N_GT = len(gt_relations)   # No. of ground-truth objects

        if N_GT == 0:
            # No ground-truth relations to match
            return torch.empty((0, N_R), dtype=torch.int64, device=gt_relations.device)

        # Mapping from proposal sample indices (pairs) to matched ground-truth object indices (pairs)
        sampled_targets = matched_idxs[sampled_idxs]
        sampled_target_pairs = torch.stack(
            [sampled_targets[:,None].expand(N_S, N_S), sampled_targets[None,:].expand(N_S, N_S)],
            dim=2
        )
        sampled_target_pairs = sampled_target_pairs[:,:,0] * N_GT + sampled_target_pairs[:,:,1]
        sampled_target_pairs = sampled_target_pairs.view(-1)[:,None]

        # Ground truth relation annotations by sampled proposal indices
        gt_rels_by_smp = torch.gather(gt_relations.view(-1, N_R), 0, sampled_target_pairs.expand(-1, N_R))
        # Don't convert its view from (N_S**2, N_R) to (N_S, N_S, N_R)...

        # Ground truth relation annotations for the sampled proposal index pairs
        sampled_idx_pairs = sampled_idx_pairs[:,0] * N_S + sampled_idx_pairs[:,1]
        sampled_idx_pairs = sampled_idx_pairs[:,None]
        gt_rels_by_smp_pairs = torch.gather(gt_rels_by_smp, 0, sampled_idx_pairs.expand(-1, N_R))

        return gt_rels_by_smp_pairs
    
    def _compute_pair_enclosing_boxes(self, proposals, sampled_idx_pairs):
        """
        Helper method to compute bounding boxes that enclose designated pairs of sampled proposals,
        according to sampled_idx_pairs
        """
        if len(sampled_idx_pairs) == 0:
            return Boxes([]).to(sampled_idx_pairs.device)
        else:
            boxes1 = proposals[sampled_idx_pairs[:,0]].proposal_boxes.tensor
            boxes2 = proposals[sampled_idx_pairs[:,1]].proposal_boxes.tensor

            boxes1_mins = boxes1[:,:2]; boxes2_mins = boxes2[:,:2]
            boxes1_maxs = boxes1[:,2:]; boxes2_maxs = boxes2[:,2:]

            pair_mins = torch.min(boxes1_mins, boxes2_mins)
            pair_maxs = torch.max(boxes1_maxs, boxes2_maxs)
            pair_boxes = cat([pair_mins, pair_maxs], dim=-1)

            return Boxes(pair_boxes)

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Extended to return Instances for (sampled) pairs of proposals, along with corresponding
        indices and ground-truth relation annotations, in addition to attaching ground-truth
        attribute annotations
        """
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []
        proposal_pairs_with_gt = []

        num_fg_samples = []; num_bg_samples = []
        num_pr_samples = []; num_nr_samples = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes, gt_attributes = self._sample_proposals(
                matched_idxs, matched_labels,
                targets_per_image.gt_classes, targets_per_image.gt_attributes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_attributes = gt_attributes

            # Sample pairs of sampled proposal indices we will use for training the relation
            # prediction components
            sampled_idx_pairs = self._sample_proposal_pairs(
                matched_idxs, matched_labels, sampled_idxs, targets_per_image.gt_relations
            )

            # Compute reformatted ground-truth relation annotations so that they comply
            # with the sampling indices
            gt_relations = self._fetch_rels_for_sampled_pairs(
                matched_idxs, sampled_idxs, sampled_idx_pairs, targets_per_image.gt_relations
            )

            # Get bounding boxes that enclose the sampled pairs of proposals, for computing
            # visual features needed for relation prediction
            proposal_pair_boxes = self._compute_pair_enclosing_boxes(
                proposals_per_image, sampled_idx_pairs
            )

            proposal_pairs_per_image = Instances(proposals_per_image.image_size)
            proposal_pairs_per_image.proposal_pair_idxs = sampled_idx_pairs
            proposal_pairs_per_image.proposal_pair_boxes = proposal_pair_boxes
            proposal_pairs_per_image.gt_relations = gt_relations

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append(float((gt_classes[:,0] == self.num_classes).sum()))
            num_fg_samples.append(len(gt_classes) - num_bg_samples[-1])
            num_nr_samples.append(float((gt_relations.sum(dim=1) == 0).sum()))
            num_pr_samples.append(len(gt_relations) - num_nr_samples[-1])
            proposals_with_gt.append(proposals_per_image)
            proposal_pairs_with_gt.append(proposal_pairs_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_pr_samples", np.mean(num_pr_samples))
        storage.put_scalar("roi_head/num_nr_samples", np.mean(num_nr_samples))

        return proposals_with_gt, proposal_pairs_with_gt

    def add_proposal_pairs(self, proposals):
        """
        Helper method to create and return an Instances object containing pairs of proposals
        """
        proposal_pairs = []
        for proposals_per_image in proposals:
            boxes = proposals_per_image.proposal_boxes.tensor
            box_pairs = pair_vals(boxes)

            boxes1_mins = box_pairs[:,0,:2]; boxes2_mins = box_pairs[:,1,:2]
            boxes1_maxs = box_pairs[:,0,2:]; boxes2_maxs = box_pairs[:,1,2:]

            pair_mins = torch.min(boxes1_mins, boxes2_mins)
            pair_maxs = torch.max(boxes1_maxs, boxes2_maxs)
            pair_boxes = cat([pair_mins, pair_maxs], dim=-1)

            proposal_pairs_per_image = Instances(proposals_per_image.image_size)
            proposal_pairs_per_image.proposal_pair_idxs = pair_vals(torch.arange(len(boxes))).to(boxes.device)
            proposal_pairs_per_image.proposal_pair_boxes = Boxes(pair_boxes)

            proposal_pairs.append(proposal_pairs_per_image)

        return (proposals, proposal_pairs)

    def _forward_box(self, features, proposals, boxes_provided=False):
        """
        Forward logic extended to pass RoI-pooled features extracted from pair-enclosing
        boxes to the output layer as well
        """
        proposals_objs, proposals_rels = proposals

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals_objs])
        box_features = self.box_head(box_features)

        box_pair_features = self.box_pooler(features, [x.proposal_pair_boxes for x in proposals_rels])
        box_pair_features = self.box_head(box_pair_features)

        predictions = self.box_predictor(box_features, box_pair_features, proposals)
        del box_features, box_pair_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals, boxes_provided=boxes_provided
            )
            return pred_instances

    def forward(self, images, features, proposals, targets=None, boxes_provided=False):
        """
        Minimally extended to add proposal pair preparation logic in inference mode
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        else:
            proposals = self.add_proposal_pairs(proposals)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, boxes_provided=boxes_provided)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, boxes_provided=boxes_provided)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
