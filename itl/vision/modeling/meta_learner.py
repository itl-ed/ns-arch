"""
Implements a meta-learner module that learns how to generate "predicate code" (i.e.
embeddings of class/attribute/relation categories that parametrize the box classifiers
in the scene graph generation model) from few examples.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.events import get_event_storage
from fvcore.nn import smooth_l1_loss


EPS = 1e-10           # Value used for numerical stabilization

class MetaLearner(nn.Module):

    def __init__(self, code_size, loss_type):
        """
        Args:
            code_size: int; size of category code (predicate embedding) parametrizing each
                box category classifier in base model box predictor, and also the size of
                category codes to be few-shot generated from exemplars
        """
        super().__init__()

        self.loss_type = loss_type

        self.relu = nn.ReLU()

        # Code generators for class/attribute/relation categories, implemented as two
        # bottlenecked fully-connected layers
        self.cls_code_gen = nn.Sequential(
            nn.Linear(code_size, code_size // 2),
            self.relu,
            nn.Linear(code_size // 2, code_size)
        )
        self.att_code_gen = nn.Sequential(
            nn.Linear(code_size, code_size // 2),
            self.relu,
            nn.Linear(code_size // 2, code_size)
        )
        self.rel_code_gen = nn.Sequential(
            nn.Linear(code_size, code_size // 2),
            self.relu,
            nn.Linear(code_size // 2, code_size)
        )

    def forward(self, episodes, base_model):
        """
        Args:
            episodes: List[Dict[str, Dict[int, Dict]]]; Few-shot learning episodes for each
                of class/attribute/relation recognition, sampled by FewShotDataset and mapped
                by VGFewShotMapper
            base_model: DualModeRCNN; base scene graph generation model instance, most likely
                sufficiently pretrained, with weights appropriately frozen
        
        Returns:
            Either loss values or inference results, depending on self.training
        """
        if self.training:
            return self.losses(episodes, base_model)
        else:
            return self.inference(episodes, base_model)

    def losses(self, episodes, base_model):
        """
        Args:
            (Same as self.forward())
        
        Returns:
            Dict[str, torch.tensor]; dict instance containing loss values (NCA, regularization)
        """
        rh = base_model.roi_heads    # Shortcut for brevity

        losses = {
            "loss_fs_cls": 0, "loss_reg_cls": 0,
            "loss_fs_att": 0, "loss_reg_att": 0,
            "loss_fs_rel": 0, "loss_reg_rel": 0,
        }

        for epi in episodes:
            K = epi.pop("shot")
            for cat_type in ["cls", "att", "rel"]:
                # Compute features and codes as needed
                process_fn = getattr(self, f"_process_episode_{cat_type}")
                features, codes, cats = process_fn(epi[cat_type], base_model)

                N = len(cats)
                cats_cat = torch.cat([
                    torch.tensor(cs, device=base_model.device) for cs in cats
                ], dim=0)

                if self.loss_type == "nca":
                    # Neighborhood componenet analysis (NCA) loss

                    # Dot product (code.feature) for every instance pair
                    sims = (codes[:,None,:] * self.relu(features)[None,:,:]).sum(dim=-1)

                    # Ground truth category agreement
                    gt_match = cats_cat[:,None] == cats_cat[None,:]

                else:
                    # More traditional "support vs. query" training loss
                    assert self.loss_type == "sq", "Invalid loss type"
                    
                    Is = [len(cs) for cs in cats]
                    Qs = [Ii-K for Ii in Is]
                    I_slices = np.cumsum([0]+Is)

                    # Category codes from support set, features from query set
                    S_codes = [codes[I_slices[c]:I_slices[c+1]] for c in range(N)]
                    S_codes = torch.stack([c[:K].mean(dim=0) for c in S_codes], dim=0)

                    Q_features = [features[I_slices[c]:I_slices[c+1]] for c in range(N)]
                    Q_features = torch.cat([f[K:] for f in Q_features], dim=0)

                    # Dot product (code.feature) between extracted codes vs. query instances
                    sims = (S_codes[:,None,:] * self.relu(Q_features)[None,:,:]).sum(dim=-1)

                    gt_match = torch.stack([
                        F.pad(torch.ones(Qs[c]), (sum(Qs[:c]), sum(Qs[c+1:])))
                        for c in range(N)
                    ], dim=0).to(base_model.device)
                
                losses[f"loss_fs_{cat_type}"] = \
                    losses[f"loss_fs_{cat_type}"] + F.binary_cross_entropy_with_logits(
                        sims,
                        torch.clamp(gt_match.to(sims.dtype), min=EPS, max=1-EPS),
                        reduction="mean",
                        pos_weight=torch.tensor([N-1], device=base_model.device)
                    )

                # Regularize with batch-learned class codes
                losses[f"loss_reg_{cat_type}"] = \
                    losses[f"loss_reg_{cat_type}"] + smooth_l1_loss(
                        codes,
                        getattr(rh.box_predictor, f"{cat_type}_codes").weight[cats_cat],
                        rh.box_predictor.smooth_l1_beta,
                        reduction="mean"
                    )
        
                _log_classification_stats(sims, gt_match, cat_type)

        return {k: v / len(episodes) for k, v in losses.items()}
    
    def inference(self, episodes, base_model):
        """
        Args:
            (Same as self.forward())
        
        Returns:
            Dict[str, torch.tensor]; dict instance containing N-way K-shot prediction scores
        """
        results = []

        for epi in episodes:
            K = epi.pop("shot")

            for cat_type in epi:
                # Compute features and codes as needed
                process_fn = getattr(self, f"_process_episode_{cat_type}")
                features, codes, cats = process_fn(epi[cat_type], base_model, batchwise=False)

                # Split by categories; here we are being safe by not assuming that the categories
                # all have the same number of instances (we use same support set size, but query
                # set sizes may be different due to filtering in mapper)
                N = len(cats)
                Is = [len(cs) for cs in cats]
                Qs = [Ii-K for Ii in Is]
                I_slices = np.cumsum([0]+Is)

                features = [features[I_slices[c]:I_slices[c+1]] for c in range(N)]
                codes = [codes[I_slices[c]:I_slices[c+1]] for c in range(N)]

                # Category codes from support set, features from query set
                S_codes = torch.stack([c[:K].mean(dim=0) for c in codes], dim=0)
                Q_features = torch.cat([f[K:] for f in features], dim=0)

                # Dot product (code.feature) between extracted codes vs. query instances
                sims = (S_codes[:,None,:] * self.relu(Q_features)[None,:,:]).sum(dim=-1)
                sims = torch.sigmoid(sims)

                gt_match = torch.stack([
                    F.pad(torch.ones(Qs[c]), (sum(Qs[:c]), sum(Qs[c+1:])))
                    for c in range(N)
                ], dim=0)

                assert sims.shape == gt_match.shape

                results.append({
                    cat_type: {
                        "scores": sims,
                        "ground_truth": gt_match,
                        "categories": [c[0] for c in cats]
                    }
                })

        return results

    def _process_episode_cls(self, epi_cls, base_model, batchwise=True):
        """
        Helper method that carries out necessary computations given a few-shot class
        recognition episode

        Args:
            epi_cls: Dict[int, List[Dict]]; Few-shot learning episode, consisting of I class
                instances for each of the N classes
            base_model: torch.nn.Module; Base vision model passed to self.forward()
            batchwise: bool; Set as False to allow instance-wise image processing, so that
                images in a batch are not collectively preprocessed to yield a padded batch.
                Needed when images are not randomly cropped (as during training) and thus padded
                batches would get too large.
        
        Returns:
            cls_features: torch.tensor (N*D); Prediction-ready box feature for each instance
            cls_codes: torch.tensor (N*D); Class code generated by this meta learner from each
                instance
            cls_cats: torch.tensor (N); Class category indices for each instance
        """
        rh = base_model.roi_heads

        # Compute box features for the instances with the base_model
        cls_features = []
        for c, insts in epi_cls.items():
            proposals_objs = [ins["proposals"].to(base_model.device) for ins in insts]

            if batchwise:
                img_features = base_model.backbone(base_model.preprocess_image(insts).tensor)
                img_features = [img_features[f] for f in rh.box_in_features]

                box_features = rh.box_pooler(
                    img_features, [x.proposal_boxes for x in proposals_objs]
                )
                box_features = rh.box_head(box_features)
            else:
                img_features = [
                    base_model.backbone(base_model.preprocess_image([ins]).tensor)
                    for ins in insts
                ]
                img_features = [
                    [ins[f] for f in rh.box_in_features]
                    for ins in img_features
                ]

                box_features = torch.cat([
                    rh.box_pooler(ins_f, [ins_p.proposal_boxes])
                    for ins_f, ins_p in zip(img_features, proposals_objs)
                ], dim=0)
                box_features = rh.box_head(box_features)

            cls_f, _, _ = rh.box_predictor(
                box_features, None, (proposals_objs, None), return_features=True
            )

            del box_features
            for ins in insts: del ins["image"]

            cls_features.append((c, cls_f))
        
        # Generate codes from each instance
        cls_codes = [(c, self.cls_code_gen(f)) for c, f in cls_features]

        cls_cats = [[c]*len(f) for c, f in cls_features]

        cls_features = torch.cat([f for _, f in cls_features], dim=0)
        cls_codes = torch.cat([code for _, code in cls_codes], dim=0)

        return cls_features, cls_codes, cls_cats

    def _process_episode_att(self, epi_att, base_model, batchwise=True):
        """
        Helper method analogous to self._process_episode_cls, but for few-shot attribute
        recognition task instead
        """
        rh = base_model.roi_heads

        att_features = []
        for a, insts in epi_att.items():
            proposals_objs = [ins["proposals"].to(base_model.device) for ins in insts]

            if batchwise:
                img_features = base_model.backbone(base_model.preprocess_image(insts).tensor)
                img_features = [img_features[f] for f in rh.box_in_features]

                box_features = rh.box_pooler(
                    img_features, [x.proposal_boxes for x in proposals_objs]
                )
                box_features = rh.box_head(box_features)
            else:
                img_features = [
                    base_model.backbone(base_model.preprocess_image([ins]).tensor)
                    for ins in insts
                ]
                img_features = [
                    [ins[f] for f in rh.box_in_features]
                    for ins in img_features
                ]

                box_features = torch.cat([
                    rh.box_pooler(ins_f, [ins_p.proposal_boxes])
                    for ins_f, ins_p in zip(img_features, proposals_objs)
                ], dim=0)
                box_features = rh.box_head(box_features)

            _, att_f, _ = rh.box_predictor(
                box_features, None, (proposals_objs, None), return_features=True
            )

            del box_features
            for ins in insts: del ins["image"]

            att_features.append((a, att_f))

        att_codes = [(a, self.att_code_gen(f)) for a, f in att_features]

        att_cats = [[a]*len(f) for a, f in att_features]

        att_features = torch.cat([f for _, f in att_features], dim=0)
        att_codes = torch.cat([code for _, code in att_codes], dim=0)

        return att_features, att_codes, att_cats

    def _process_episode_rel(self, epi_rel, base_model, batchwise=True):
        """
        Helper method analogous to self._process_episode_cls, but for few-shot relation
        recognition task instead
        """
        rh = base_model.roi_heads

        rel_features = []
        for r, insts in epi_rel.items():
            proposals = [ins["proposals"].to(base_model.device) for ins in insts]
            proposals_objs, proposals_rels = rh.add_proposal_pairs(proposals)

            if batchwise:
                img_features = base_model.backbone(base_model.preprocess_image(insts).tensor)
                img_features = [img_features[f] for f in rh.box_in_features]

                box_features = rh.box_pooler(
                    img_features, [x.proposal_boxes for x in proposals_objs]
                )
                box_features = rh.box_head(box_features)

                # Need to compute & feed box pair features as well, contrary to the counterparts
                # for cls/att episode processing methods
                box_pair_features = rh.box_pooler(
                    img_features, [x.proposal_pair_boxes for x in proposals_rels]
                )
                box_pair_features = rh.box_head(box_pair_features)
            else:
                img_features = [
                    base_model.backbone(base_model.preprocess_image([ins]).tensor)
                    for ins in insts
                ]
                img_features = [
                    [ins[f] for f in rh.box_in_features]
                    for ins in img_features
                ]

                box_features = torch.cat([
                    rh.box_pooler(ins_f, [ins_p.proposal_boxes])
                    for ins_f, ins_p in zip(img_features, proposals_objs)
                ], dim=0)
                box_features = rh.box_head(box_features)

                box_pair_features = torch.cat([
                    rh.box_pooler(ins_f, [ins_p.proposal_pair_boxes])
                    for ins_f, ins_p in zip(img_features, proposals_rels)
                ], dim=0)
                box_pair_features = rh.box_head(box_pair_features)

            # Provide all the necessary arguments
            _, _, rel_f = rh.box_predictor(
                box_features, box_pair_features, (proposals_objs, proposals_rels),
                return_features=True
            )

            # We only need the instance pairs indexed (0,1), so discard every other row
            # to leave only the odd-numbered rows
            rel_f = rel_f[0::2]

            del box_features, box_pair_features
            for ins in insts: del ins["image"]

            rel_features.append((r, rel_f))

        rel_codes = [(r, self.rel_code_gen(f)) for r, f in rel_features]

        rel_cats = [[r]*len(f) for r, f in rel_features]

        rel_features = torch.cat([f for _, f in rel_features], dim=0)
        rel_codes = torch.cat([code for _, code in rel_codes], dim=0)

        return rel_features, rel_codes, rel_cats

def _log_classification_stats(sims, gt_match, prefix):
    """
    Log the few-shot classification metrics to EventStorage.

    Args:
        sims: torch.tensor (N*N); Code-feature agreement logits
        gt_classes: torch.tensor (N*N); Ground-truth category agreement labels
        prefix: str; Classification metric name prefix
    """
    storage = get_event_storage()

    num_match = gt_match.sum()

    # Prediction by binary thresholding (instead of argmax, as labels are not exclusive)
    thresholds = [0.5, 0.8]
    threshold_logits = torch.tensor(thresholds, device=sims.device).logit()

    pred_match = sims[..., None] > threshold_logits[None, None, :]
    
    for i, thresh in enumerate(thresholds):
        AP = pred_match[..., i].sum()
        TP = torch.logical_and(pred_match[..., i].squeeze(), gt_match).sum()

        if AP > 0:
            storage.put_scalar(f"meta_learner/{prefix}_P@{thresh}", TP / AP)
        if num_match > 0:
            storage.put_scalar(f"meta_learner/{prefix}_R@{thresh}", TP / num_match)
            storage.put_scalar(f"meta_learner/{prefix}_PRatio@{thresh}", AP / num_match)
