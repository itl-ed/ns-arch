"""
Implement custom modules by extending detectron2-provided defaults, to be plugged
into our scene graph generation model
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from detectron2.layers import cat, batched_nms
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.utils.events import get_event_storage

from ..utils import pair_select_indices


EPS = 1e-10           # Value used for numerical stabilization
ROI_HEAD_WGT = 10     # Weight multiplier for all losses computed in this predictor

class SceneGraphRCNNOutputLayers(FastRCNNOutputLayers):
    """
    In addition to the two per-class linear layers for box regression and class
    prediction, need to add per-attribute/relation prediction layers

    Re-implement:
        __init__(), from_config(), forward(), losses(), inference(), predict_boxes(), predict_probs()
    """

    # (Max) size of category code (predicate embedding) parametrizing each box category
    # classifier head
    CODE_SIZE = 256

    @configurable
    def __init__(
        self,
        *,
        num_attributes,
        num_relations,
        cfg,
        **kwargs
    ):
        """
        Add info about numbers of attribute/relation predicates as well, while splitting
        class prediction layer into two
        """
        super().__init__(**kwargs)
        self.num_attributes = num_attributes
        self.num_relations = num_relations
        self.cfg = cfg        # Needed for loading positive loss weights

        # Dimensionality reduction layers
        input_size = self.cls_score.in_features            # Inherited from parent class...
        input_size_cmp = min(input_size, self.CODE_SIZE)   # ... and compressed down to CODE_SIZE, if larger

        delattr(self, "cls_score")                         # No longer needed

        self.compress_cls = nn.Linear(input_size, input_size_cmp)
        self.compress_att = nn.Linear(input_size+input_size_cmp, input_size_cmp)
        self.compress_rel = nn.Linear(input_size+22+input_size_cmp*2, input_size_cmp)

        # Split self.cls_score into two nn.Linear layers, each responsible for class
        # prediction and objectness prediction, for more flexible registration of new
        # classes
        self.objectness_score = nn.Linear(input_size, 1)
        self.cls_codes = nn.Linear(input_size_cmp, self.num_classes, bias=False)
        self.att_codes = nn.Linear(input_size_cmp, self.num_attributes, bias=False)
        self.rel_codes = nn.Linear(input_size_cmp, self.num_relations, bias=False)

        self.relu = nn.ReLU()
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        """
        Add info about numbers of attribute/relation predicates as well
        """
        ret = super().from_config(cfg, input_shape)
        ret["num_attributes"] = cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES
        ret["num_relations"] = cfg.MODEL.ROI_HEADS.NUM_RELATIONS
        ret["cfg"] = cfg

        return ret

    def _pair_visual_features(self, box_f, box_pair_f, arg1_idxs, arg2_idxs):
        """
        Helper method for computing visual features for pairs of boxes
        """
        return box_pair_f - \
            torch.gather(box_f, 0, arg1_idxs.expand(-1, box_f.shape[-1])) - \
            torch.gather(box_f, 0, arg2_idxs.expand(-1, box_f.shape[-1]))
    
    def _pair_spatial_features(self, boxes, box_pairs, arg1_idxs, arg2_idxs):
        """
        Helper method for computing spatial features for pairs of boxes
        """
        boxes_xyxy = boxes.tensor                       # min/max coordinates normalized by image size
        boxes_wh = boxes_xyxy[:,2:] - boxes_xyxy[:,:2]  # widths and heights (normlized)
        boxes_c = boxes.get_centers()                   # center points (normlized)
        boxes_A = boxes_wh[:,0] * boxes_wh[:,1]         # box areas (normlized)

        # Coordinate features
        boxes_xyxy_arg1s = torch.gather(boxes_xyxy, 0, arg1_idxs.expand(-1, 4))
        boxes_xyxy_arg2s = torch.gather(boxes_xyxy, 0, arg2_idxs.expand(-1, 4))
        coord_f = cat([
            boxes_xyxy_arg1s, boxes_xyxy_arg2s,
            torch.gather(boxes_wh, 0, arg1_idxs.expand(-1, 2)) / \
                torch.gather(boxes_wh, 0, arg2_idxs.expand(-1, 2))
        ], dim=-1)

        # Distance features
        dist_f = cat([
            coord_f[:,4:8] - coord_f[:,:4],
            torch.gather(boxes_c, 0, arg2_idxs.expand(-1, 2)) / \
                torch.gather(boxes_c, 0, arg1_idxs.expand(-1, 2))
        ], dim=-1)

        # Area features
        boxes_A_arg1s = torch.gather(boxes_A[:,None], 0, arg1_idxs)
        boxes_A_arg2s = torch.gather(boxes_A[:,None], 0, arg2_idxs)
        intersections_xyxy = cat([
            torch.max(boxes_xyxy_arg1s[:,:2], boxes_xyxy_arg2s[:,:2]),
            torch.min(boxes_xyxy_arg1s[:,2:], boxes_xyxy_arg2s[:,2:])
        ], dim=-1)
        intersections_wh = intersections_xyxy[:,2:] - intersections_xyxy[:,:2]
        intersections_A = intersections_wh[:,0] * intersections_wh[:,1]
        intersections_A *= (intersections_wh[:,0] >= 0) * (intersections_wh[:,1] >= 0)
        intersections_A = intersections_A[:,None]
        unions_A = boxes_A_arg1s + boxes_A_arg2s - intersections_A
        box_pair_supports_A = box_pairs.area()
        box_pair_supports_A = box_pair_supports_A[:,None]

        area_f = cat([
            boxes_A_arg1s, boxes_A_arg2s,
            boxes_A_arg1s / boxes_A_arg2s,
            intersections_A / boxes_A_arg1s,
            intersections_A / boxes_A_arg2s,
            unions_A / box_pair_supports_A
        ], dim=-1)

        return cat([coord_f, dist_f, area_f], dim=-1)
    
    def _pair_semantic_features(self, sem_f, arg1_idxs, arg2_idxs):
        """
        Helper method for computing semantic features for pairs of boxes
        """
        sem_f_arg1s = torch.gather(sem_f, 0, arg1_idxs.expand(-1, sem_f.shape[-1]))
        sem_f_arg2s = torch.gather(sem_f, 0, arg2_idxs.expand(-1, sem_f.shape[-1]))

        return cat([sem_f_arg1s, sem_f_arg2s], dim=-1)
    
    def forward(self, box_features, box_pair_features, proposals):
        """
        Extended for a new interface --

        Args:
            box_features: Tensor; per-region features of shape (N_R, ...) for N_R bounding boxes
                to predict
            box_pair_features: Tensor; per-region features of shape (N_P, ...) for N_P pair-enclosing
                boxes to predict -- if None, skip relation prediction
            proposals: (Instances, Instances), passed from super().forward(); needed for spatial
                feature extraction

        Returns:
            (Tensor, Tensor, Tensor, Tensor, Tensor):
                First tensor: shape (N_R, K_C), scores for each of the N_R boxes. Each row contains the
                scores for K_C object classes.

                Second tensor: shape (N_R, 1), scores for each of the N_R boxes. Each row contains the
                scores for whether the box contains an object (i.e. is foreground class) or not.

                Third tensor: bounding box regression deltas for each box. Shape is shape (N_R, K_C*4),
                or (N_R, 4) for class-agnostic regression.

                Fourth tensor: shape (N_R, K_A), scores for each of the N_R boxes. Each row contains the
                scores for K_A object attributes.

                Fifth tensor: shape (N_P, K_R), scores for each of the N_P box pairs. Each row contains
                the scores for K_R object pair relations.
            
            (Tensor, Tensor, Tensor, Tensor, Tensor):
                shape (N_R/N_R/N_P, D), feature vectors before object class/attribute/relation prediction
                shape (N_R, D), semantic feature vectors
        """
        if box_features.dim() > 2:
            box_features = torch.flatten(box_features, start_dim=1)
        if (box_pair_features is not None) and (box_pair_features.dim() > 2):
            box_pair_features = torch.flatten(box_pair_features, start_dim=1)

        proposals_objs, proposals_rels = proposals

        # Compute class and objectness prediction scores separately, for flexible addition of novel
        # class concepts...
        objectness_scores = self.objectness_score(box_features)

        cls_compressed_f = self.compress_cls(box_features)
        cls_scores = self.cls_codes(self.relu(cls_compressed_f))

        # Aggregated 'semantic' feature for each box for later use, weighted by class probabilities.
        # Per-class features in the class prediction layer are used as semantic features

        # Use estimated class category probabilities
        assert cls_scores.shape[-1] == self.num_classes
        cls_probs = torch.sigmoid(cls_scores)        
        if self.training:
            # Add ground-truth class category labels as additional weights
            cls_probs = cls_probs + cat([pos.gt_classes for pos in proposals_objs]).to(torch.float)
            cls_probs = cls_probs / 2

        sem_features = F.linear(cls_probs, self.cls_codes.weight.T)
        sem_features = sem_features / (cls_probs.sum(dim=1, keepdim=True)+EPS)

        # Bbox deltas are computed in the same way
        proposal_deltas = self.bbox_pred(box_features)

        ## Attribute prediction, using following features
        #   1) Visual features (use as-is)
        #   2) Semantic features (i.e. 'taking into account the predicted classes')
        att_compressed_f = self.compress_att(cat([box_features, sem_features], dim=-1))
        att_scores = self.att_codes(self.relu(att_compressed_f))

        # Skip relation prediction if box pair features is not provided
        if box_pair_features is None:
            rel_compressed_f = None
            rel_scores = None
        else:
            ## Relation prediction, using following features (Consult Plesse et al., 2020 for details):
            #   1) Visual features
            #   2) Spatial features
            #   3) Semantic features

            # Shift box indices by corresponding numbers of instances in each image in the batch
            shifts = [0] + [len(pos) for pos in proposals_objs][:-1]
            shifts = torch.tensor(shifts, device=box_features.device).cumsum(dim=0)
            pair_idxs_shifted = [prs.proposal_pair_idxs + s for prs, s in zip(proposals_rels, shifts)]
            pair_idxs_shifted = cat(pair_idxs_shifted, dim=0)

            # arg1 & arg2 indices to be used by torch.gather in pair feature computations
            arg1_idxs_shifted = pair_idxs_shifted[:, 0, None]
            arg2_idxs_shifted = pair_idxs_shifted[:, 1, None]

            # Normalize box dimensions by image width/heights
            for pos in proposals_objs:
                bxs = pos.proposal_boxes
                h_I, w_I = pos.image_size
                bxs.scale(1 / w_I, 1 / h_I)
            for prs in proposals_rels:
                bxs = prs.proposal_pair_boxes
                h_I, w_I = pos.image_size
                bxs.scale(1 / w_I, 1 / h_I)

            # 1) Visual features
            pair_vis_f = self._pair_visual_features(
                box_features, box_pair_features,
                arg1_idxs_shifted, arg2_idxs_shifted
            )

            # 2) Spatial features
            pair_spt_f = self._pair_spatial_features(
                Boxes.cat([pos.proposal_boxes for pos in proposals_objs]),
                Boxes.cat([pos.proposal_pair_boxes for pos in proposals_rels]),
                arg1_idxs_shifted, arg2_idxs_shifted
            )

            # 3) Semantic features
            pair_sem_f = self._pair_semantic_features(
                sem_features, arg1_idxs_shifted, arg2_idxs_shifted
            )

            rel_compressed_f = self.compress_rel(cat([pair_vis_f, pair_spt_f, pair_sem_f], dim=-1))
            rel_scores = self.rel_codes(self.relu(rel_compressed_f))

            # De_normalize box dimensions
            for pos in proposals_objs:
                bxs = pos.proposal_boxes
                h_I, w_I = pos.image_size
                bxs.scale(w_I, h_I)
            for prs in proposals_rels:
                bxs = prs.proposal_pair_boxes
                h_I, w_I = pos.image_size
                bxs.scale(w_I, h_I)

        out = (
            objectness_scores,
            cls_scores, att_scores, rel_scores,
            proposal_deltas
        )
        f_vecs = cls_compressed_f, att_compressed_f, rel_compressed_f, sem_features

        return out, f_vecs

    def forward_inc_rels(self, box_features, box_pair_features, proposals):
        """
        Variant of self.forward(), called for incremental prediction of relations between pairs
        of newly added detections and existing detections (i.e. top right and bottom left parts
        of the 2-by-2 partitioning of the incrementally updated scene graph matrix)

        Args:
            box_features: Tensor; per-region features of shape (N_R, ...) for N_R bounding boxes
                to predict
            box_pair_features: Tensor; per-region features of shape (N_P, ...) for N_P pair-enclosing
                boxes to predict -- if None, skip relation prediction
            proposals: (Instances, Instances), passed from super().forward(); needed for spatial
                feature extraction

        Returns:
            Tensor:
                shape (N_P, K_R), scores for each of the N_P box pairs. Each row contains
                the scores for K_R object pair relations.
            
            Tensor:
                shape (N_P, D), feature vectors before object relation prediction
        """
        if box_features.dim() > 2:
            box_features = torch.flatten(box_features, start_dim=1)
        if (box_pair_features is not None) and (box_pair_features.dim() > 2):
            box_pair_features = torch.flatten(box_pair_features, start_dim=1)

        proposals_objs, proposals_rels = proposals

        ## Relation prediction

        # arg1 & arg2 indices to be used by torch.gather in pair feature computations
        arg1_idxs = proposals_rels.proposal_pair_idxs[:, 0, None]
        arg2_idxs = proposals_rels.proposal_pair_idxs[:, 1, None]

        # Normalize box dimensions by image width/heights
        bxs = proposals_objs.proposal_boxes
        h_I, w_I = proposals_objs.image_size
        bxs.scale(1 / w_I, 1 / h_I)

        bxs = proposals_rels.proposal_pair_boxes
        h_I, w_I = proposals_rels.image_size
        bxs.scale(1 / w_I, 1 / h_I)

        # 1) Visual features
        pair_vis_f = self._pair_visual_features(
            box_features, box_pair_features, arg1_idxs, arg2_idxs
        )

        # 2) Spatial features
        pair_spt_f = self._pair_spatial_features(
            proposals_objs.proposal_boxes, proposals_rels.proposal_pair_boxes,
            arg1_idxs, arg2_idxs
        )

        # 3) Semantic features
        pair_sem_f = self._pair_semantic_features(
            proposals_objs.sem_f_vecs, arg1_idxs, arg2_idxs
        )

        rel_compressed_f = self.compress_rel(cat([pair_vis_f, pair_spt_f, pair_sem_f], dim=-1))
        rel_scores = self.rel_codes(self.relu(rel_compressed_f))

        # De_normalize box dimensions
        bxs = proposals_objs.proposal_boxes
        h_I, w_I = proposals_objs.image_size
        bxs.scale(w_I, h_I)

        bxs = proposals_rels.proposal_pair_boxes
        h_I, w_I = proposals_rels.image_size
        bxs.scale(w_I, h_I)

        return rel_scores, rel_compressed_f

    def losses(self, predictions, proposals):
        """
        Extended for a new interface --

        Args:
            predictions: (Tensor, Tensor, Tensor, Tensor, Tensor), return value of self.forward()
            proposals: (Instances, Instances), passed from super().forward()
        
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        objectness_scores = predictions[0]
        cls_scores = predictions[1]
        att_scores = predictions[2]
        rel_scores = predictions[3]
        proposal_deltas = predictions[4]

        proposals_objs, proposals_rels = proposals

        # package prediction ground-truth prediction targets
        gt_classes = cat([p.gt_classes for p in proposals_objs], dim=0) \
            if len(proposals) else torch.empty(0)
        gt_attributes = cat([p.gt_attributes for p in proposals_objs], dim=0) \
            if len(proposals) else torch.empty(0)
        gt_relations = cat([p.gt_relations for p in proposals_rels], dim=0) \
            if len(proposals) else torch.empty(0)

        # Foreground object indices
        gt_fg_inds = gt_classes[:,0] != self.num_classes

        # Extract prediction scores & ground truth annotations for foreground objects only
        fg_cls_scores = cls_scores[gt_fg_inds]
        fg_att_scores = att_scores[gt_fg_inds]
        fg_gt_classes = gt_classes[gt_fg_inds]
        fg_gt_attributes = gt_attributes[gt_fg_inds]

        _log_classification_stats(
            objectness_scores, gt_fg_inds,
            fg_cls_scores, fg_gt_classes,
            fg_att_scores, fg_gt_attributes,
            rel_scores, gt_relations
        )

        # parse box regression outputs
        if len(proposals_objs):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals_objs], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat([
                (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                for p in proposals_objs
            ], dim=0)
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        # Prepare appropriate pos_weight values from cfg
        md = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        weights = {
            "cls": 1 / (torch.tensor(md.classes_counts)[None,:] / md.obj_counts),
            "att": 1 / (torch.tensor(md.attributes_counts)[None,:] / md.obj_counts),
            "rel": 1 / (torch.tensor(md.relations_counts)[None,:] / md.obj_pair_counts)
        }

        p = self.cfg.MODEL.ROI_HEADS.WEIGHT_EXPONENT
        M = self.cfg.MODEL.ROI_HEADS.WEIGHT_TARGET_MEAN
        scaling_factors = {
            cat_type: inv_freqs.shape[-1] * M / inv_freqs.pow(p).sum()
            for cat_type, inv_freqs in weights.items()
        }
        weights = {
            cat_type: scaling_factors[cat_type] * inv_freqs.pow(p)
            for cat_type, inv_freqs in weights.items()
        }
        weights["obj"] = 1 / torch.tensor(self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)[None,None]

        for wtype in weights:
            weights[wtype] = weights[wtype].to(proposal_boxes.device)

        # Just a template for shorter code
        wgt_bce_loss = lambda s, g, wn: F.binary_cross_entropy_with_logits(
            s, torch.clamp(g.to(s.dtype), min=EPS, max=1-EPS),
            reduction="mean", pos_weight=weights[wn]
        )

        losses = {
            "loss_obj": wgt_bce_loss(objectness_scores, gt_fg_inds[:,None], "obj"),
            "loss_cls": wgt_bce_loss(fg_cls_scores, fg_gt_classes, "cls"),
            "loss_att": wgt_bce_loss(fg_att_scores, fg_gt_attributes, "att"),
            "loss_rel": wgt_bce_loss(rel_scores, gt_relations, "rel"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes[:,0]
            ),
        }

        return {k: v * self.loss_weight.get(k, 1.0) * ROI_HEAD_WGT for k, v in losses.items()}

    def inference(self, predictions, proposals, boxes_provided=False, nms_scores=None):
        """
        Extended for a new scene graph generation inference output format; the method effectively
        fuses super().inference(), fast_rcnn_inference(), and fast_rcnn_inference_single_image()
        in the original source (detectron2/modeling/roi_heads/fast_rcnn.py) into a single method

        Args:
            predictions: (Tensor, Tensor, Tensor, Tensor, Tensor), return value of self.forward()
            proposals: (Instances, Instances), passed from super().forward()
            boxes_provided: bool (optional), whether to skip filtering by objectness score & NMS
                -- expected to be True for 'classification mode' where inputs come with proposals
                from ground truth bounding boxes
            nms_scores: Tensor (optional), scores with which NMS scores will be run instead of
                the objectness scores (which is generally equivalent to salience in scene),
                expectedly obtained from some compatibility test for visual search

        Returns:
            list[Instances]: A list of N instances, one for each image in the batch, that stores
                the topk most confidence detections.
            list[Tensor]: A list of 1D tensor of length of N, each element indicates the corresponding
                boxes/scores index in [0, Ri) from the input, for image i.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals[0]]

        score_thresh = self.test_score_thresh
        nms_thresh = self.test_nms_thresh
        topk_per_image = self.test_topk_per_image

        instances = []
        kept_indices = []
        for boxes_per_image, scores_per_image, proposals_per_image, image_shape \
            in zip(boxes, scores, proposals[0], image_shapes):

            objectness_scores_per_image = scores_per_image[0]
            cls_scores_per_image = scores_per_image[1]
            att_scores_per_image = scores_per_image[2]
            rel_scores_per_image = scores_per_image[3]

            valid_mask = torch.isfinite(boxes_per_image).all(dim=1) & \
                torch.isfinite(cls_scores_per_image).all(dim=1) & \
                torch.isfinite(objectness_scores_per_image).all(dim=1) & \
                torch.isfinite(att_scores_per_image).all(dim=1)
            if not valid_mask.all():
                boxes_per_image = boxes_per_image[valid_mask]
                objectness_scores_per_image = objectness_scores_per_image[valid_mask]
                cls_scores_per_image = cls_scores_per_image[valid_mask]
                att_scores_per_image = att_scores_per_image[valid_mask]

            num_bbox_reg_classes = boxes_per_image.shape[1] // 4
            # Convert to Boxes to use the `clip` function ...
            boxes_per_image = Boxes(boxes_per_image.reshape(-1, 4))
            boxes_per_image.clip(image_shape)
            boxes_per_image = boxes_per_image.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
            if num_bbox_reg_classes == 1:
                boxes_per_image = boxes_per_image[:,0,:]

            if boxes_provided:
                filter_inds = torch.arange(len(boxes_per_image), device=boxes_per_image.device)
            else:
                if nms_scores is None:
                    ## If not given explicit measures by which proposals are to be ranked, use
                    ## objectness scores to filter and rank them

                    # First filter results based on detection scores. It can make NMS more efficient
                    # by filtering out low-confidence detections.
                    filter_mask = objectness_scores_per_image.view(len(boxes_per_image)) > score_thresh
                    filter_inds = filter_mask.nonzero().squeeze()
                    if len(filter_inds.shape) == 0:
                        filter_inds = filter_inds[None]

                    # Filter out proposal pairs containing any proposals that are filtered out
                    pair_filter_inds = pair_select_indices(len(filter_inds), filter_inds)

                    boxes_per_image = boxes_per_image[filter_mask]
                    objectness_scores_per_image = objectness_scores_per_image[filter_mask]
                    cls_scores_per_image = cls_scores_per_image[filter_mask]
                    att_scores_per_image = att_scores_per_image[filter_mask]
                    rel_scores_per_image = rel_scores_per_image[pair_filter_inds]

                    # Scores to feed into NMS
                    nms_scores = objectness_scores_per_image.view(len(filter_inds))
                else:
                    filter_inds = torch.arange(len(boxes_per_image))

                # Apply NMS to the filtered proposals, with provided scores.
                keep = batched_nms(
                    boxes_per_image, nms_scores,
                    torch.zeros(len(nms_scores), dtype=torch.int64, device=nms_scores.device),
                    nms_thresh
                )
                if topk_per_image >= 0:
                    keep = keep[:topk_per_image]

                # Indices for selecting relation prediction outputs for pairs of kept proposals
                pair_keep = pair_select_indices(len(filter_inds), keep)

                boxes_per_image = boxes_per_image[keep]
                objectness_scores_per_image = objectness_scores_per_image[keep]
                cls_scores_per_image = cls_scores_per_image[keep]
                att_scores_per_image = att_scores_per_image[keep]
                rel_scores_per_image = rel_scores_per_image[pair_keep]
                filter_inds = filter_inds[keep]

            # Reshape relation prediction outputs from 1-D flattened tensor (without diags;
            # shape (N^2 - N, R)) tensor to 2-D square tensor (shape (N, N, R))
            N = len(boxes_per_image); R = rel_scores_per_image.shape[-1]
            if N > 0:
                rel_scores_per_image = torch.stack([
                    torch.cat([
                        rel_scores_per_image[i*(N-1):i*(N-1)+i],
                        torch.zeros([1,R], device=rel_scores_per_image.device),
                        rel_scores_per_image[i*(N-1)+i:(i+1)*(N-1)]
                    ], dim=0)
                    for i in range(N)
                ], dim=0)

            result = Instances(image_shape)
            result.pred_classes = cls_scores_per_image
            result.pred_attributes = att_scores_per_image
            result.pred_relations = rel_scores_per_image
            if boxes_provided:
                result.pred_boxes = proposals_per_image.proposal_boxes
                if proposals_per_image.has("pred_objectness"):
                    result.pred_objectness = proposals_per_image.pred_objectness
                else:
                    result.pred_objectness = torch.ones_like(objectness_scores_per_image)
            else:
                result.pred_boxes = Boxes(boxes_per_image)
                result.pred_objectness = objectness_scores_per_image

            instances.append(result)
            kept_indices.append(filter_inds)

        return instances, kept_indices

    def predict_boxes(self, predictions, proposals):
        """
        Extended for a new interface --

        Args:
            predictions: (Tensor, Tensor, Tensor, Tensor, Tensor), return value of self.forward()
            proposals: (Instances, Instances), passed from super().forward()
        
        Returns:
            Same as super().predict_boxes()
        """
        return super().predict_boxes((predictions[1], predictions[4]), proposals[0])
    
    def predict_probs(self, predictions, proposals):
        """
        Extended for a new interface --

        Args:
            predictions: (Tensor, Tensor, Tensor, Tensor, Tensor), return value of self.forward()
            proposals: (Instances, Instances), passed from super().forward()
        
        Returns:
            list[(Tensor, Tensor, Tensor, Tensor)]:
            Lists of tensors, containing predicted class/objectness/attribute/relation probabilities
            for each image.
        """
        objectness_scores = predictions[0]
        cls_scores = predictions[1]
        att_scores = predictions[2]
        rel_scores = predictions[3]

        proposals_objs, proposals_rels = proposals

        num_inst_per_image = [len(p) for p in proposals_objs]
        num_inst_pair_per_image = [len(p) for p in proposals_rels]

        probs_objectness = torch.sigmoid(objectness_scores)
        probs_cls = torch.sigmoid(cls_scores)
        probs_att = torch.sigmoid(att_scores)
        probs_rel = torch.sigmoid(rel_scores)

        return list(zip(
            probs_objectness.split(num_inst_per_image, dim=0),
            probs_cls.split(num_inst_per_image, dim=0),
            probs_att.split(num_inst_per_image, dim=0),
            probs_rel.split(num_inst_pair_per_image, dim=0)
        ))


def _log_classification_stats(
        objectness_logits, gt_fg_inds,
        cls_logits, gt_classes, 
        att_logits, gt_attributes,
        rel_logits, gt_relations,
        prefix="fast_rcnn"
    ):
    """
    Extended for logging prediction results other than class predictions (multilabel)

    Args:
        objectness_logits: (N_R) logits
        cls_logits, att_logits: (N_F, [K_C,K_A]) logits
        rel_logits: (N_P, K_R) logits
        gt_fg_inds: (N_R) boolean objectness ground truths
        gt_classes, gt_attributes: (N_F, [K_C,K_A]) class/attribute ground truths 
        gt_relations: (N_P, K_R) relation ground-truths
    """
    num_instances = len(cls_logits)
    if num_instances == 0:
        return

    num_fg = gt_fg_inds.sum()

    # No. of positive class/attribute/relation annotations
    num_cls = gt_classes.sum()
    num_att = gt_attributes.sum()
    num_rel = gt_relations.sum()

    storage = get_event_storage()

    # Prediction by binary thresholding (instead of argmax, as labels are not exclusive)
    thresholds = [0.5, 0.8]
    threshold_logits = torch.tensor(thresholds, device=cls_logits.device).logit()
    threshold_logits = threshold_logits[None,:]

    pred_objectness = objectness_logits > threshold_logits
    fg_pred_classes = cls_logits[:,:,None] > threshold_logits
    fg_pred_attributes = att_logits[:,:,None] > threshold_logits
    pred_relations = rel_logits[:,:,None] > threshold_logits

    # Record scalar metrics
    for i, thresh in enumerate(thresholds):
        # Objectness precision-recall (AP here stands for 'all positives' (TP + FP))
        objectness_AP = pred_objectness[..., i].sum()
        objectness_TP = torch.logical_and(pred_objectness[..., i].squeeze(), gt_fg_inds).sum()

        if objectness_AP > 0:
            storage.put_scalar(f"{prefix}/objectness_P@{thresh}", objectness_TP / objectness_AP)
        if num_fg > 0:
            storage.put_scalar(f"{prefix}/objectness_R@{thresh}", objectness_TP / num_fg)

        # Class prediction average precision-recall
        cls_AP = fg_pred_classes[..., i].sum()
        cls_TP = torch.logical_and(fg_pred_classes[..., i], gt_classes).sum()

        if cls_AP > 0:
            storage.put_scalar(f"{prefix}/cls_AP@{thresh}", cls_TP / cls_AP)
        if num_cls > 0:
            storage.put_scalar(f"{prefix}/cls_AR@{thresh}", cls_TP / num_cls)

        # Attribute prediction average precision-recall
        att_AP = fg_pred_attributes[..., i].sum()
        att_TP = torch.logical_and(fg_pred_attributes[..., i], gt_attributes).sum()

        if att_AP > 0:
            storage.put_scalar(f"{prefix}/att_AP@{thresh}", att_TP / att_AP)
        if num_att > 0:
            storage.put_scalar(f"{prefix}/att_AR@{thresh}", att_TP / num_att)

        # Relation prediction average precision-recall
        rel_AP = pred_relations[..., i].sum()
        rel_TP = torch.logical_and(pred_relations[..., i], gt_relations).sum()

        if rel_AP > 0:
            storage.put_scalar(f"{prefix}/rel_AP@{thresh}", rel_TP / rel_AP)
        if num_rel > 0:
            storage.put_scalar(f"{prefix}/rel_AR@{thresh}", rel_TP / num_rel)
