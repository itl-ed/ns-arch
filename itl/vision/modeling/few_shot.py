"""
Helper methods factored out, which perform computations required for appropriate
exemplar-based few-shot prediction tasks (binary concept classification given
bounding boxes in image, conditioned search for bounding boxes of concept instances
in image). Methods return loss values needed for training and metric values for
evaluation.
"""
import torch
import torch.nn.functional as F

from torchvision.ops import nms
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrHungarianMatcher,
    DeformableDetrLoss,
    sigmoid_focal_loss
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def compute_loss_and_metrics(
    model, task, conc_type, detr_enc_outs, detr_dec_outs,
    B, N, K, bboxes_search_targets
):
    assert B == len(detr_dec_outs)
    assert N * K == B

    if task == "fs_classify":
        assert bboxes_search_targets is None
        return _compute_fs_classify(
            model, conc_type, detr_dec_outs, B, N, K
        )
    else:
        assert task == "fs_search"
        assert bboxes_search_targets is not None
        return _compute_fs_search(
            model, conc_type, detr_enc_outs, detr_dec_outs,
            B, N, K, bboxes_search_targets
        )


def _compute_fs_classify(model, conc_type, detr_dec_outs, B, N, K):
    """ Classification mode; compute NCA loss & recall@K metrics """
    # Pass through MLP block according to specified task concept type
    cls_embeddings = model.fs_embed_cls(detr_dec_outs[:,0,:])
    if conc_type == "classes":
        conc_embeddings = cls_embeddings
    elif conc_type == "attributes":
        conc_embeddings = model.fs_embed_att(
            torch.cat([detr_dec_outs[:,0,:], cls_embeddings], dim=-1)
        )
    else:
        raise ValueError("Invalid concept type")

    # Obtain pairwise distances (squared) between exemplars
    pairwise_dists_sq = torch.cdist(conc_embeddings, conc_embeddings, p=2.0) ** 2
    pairwise_dists_sq.fill_diagonal_(float("inf"))     # Disregarding reflexive distance

    # Compute NCA loss
    pos_dist_mask = torch.zeros_like(pairwise_dists_sq)
    for i in range(N):
        pos_dist_mask[i*K:(i+1)*K, i*K:(i+1)*K] = 1.0
    pos_dist_mask.fill_diagonal_(0.0)
    loss = F.softmax(-pairwise_dists_sq) * pos_dist_mask
    loss = -loss.sum(dim=-1).log().sum() / B

    # Compute recall@K values as performance metric
    metrics = {}
    r_at_k = 0.0; r_at_2k = 0.0
    topk_closest_exs = pairwise_dists_sq.topk(2*K, largest=False, dim=-1).indices
    for i in range(N):
        for j in range(K):
            ind = i*K + j
            gt_pos_exs = set(range(i*N, i*N+K)) - {ind}
            topk = set(topk_closest_exs[ind][:K].tolist()) - {ind}
            top2k = set(topk_closest_exs[ind].tolist()) - {ind}
            r_at_k += len(topk & gt_pos_exs) / (K - 1)
            r_at_2k += len(top2k & gt_pos_exs) / (K - 1)
    metrics["recall@K"] = r_at_k / B
    metrics["recall@2K"] = r_at_2k / B

    return loss, metrics, conc_embeddings


def _compute_fs_search(
    model, conc_type, detr_enc_outs, detr_dec_outs, B, N, K, bboxes_search_targets
):
    """ (Conditioned) search mode; compute DETR loss & mAP metrics """
    assert K > 1        # Need more than one "shots" to perform search

    # Compute needed concept embeddings, then 'leave-one-out' average prototypes
    # to be fused into search spec embeddings
    cls_embeddings = model.fs_embed_cls(detr_dec_outs[:,0,:])
    if "classes" in conc_type:
        cls_loo_protos = cls_embeddings.view(N, K, -1).sum(dim=1, keepdim=True)
        cls_loo_protos = cls_loo_protos.expand(N, K, -1).reshape(B, -1)
        cls_loo_protos = (cls_loo_protos - cls_embeddings) / (K - 1)
    else:
        # No class-related condition in search spec
        cls_loo_protos = torch.zeros(
            B, model.fs_embed_cls.layers[-1].out_features, device=model.device
        )
    if "attributes" in conc_type:
        att_embeddings = model.fs_embed_att(
            torch.cat([detr_dec_outs[:,0,:], cls_embeddings], dim=-1)
        )
        att_loo_protos = att_embeddings.view(N, K, -1).sum(dim=1, keepdim=True)
        att_loo_protos = att_loo_protos.expand(N, K, -1).reshape(B, -1)
        att_loo_protos = (att_loo_protos - att_embeddings) / (K - 1)
    else:
        # No attribute-related condition in search spec
        att_loo_protos = torch.zeros(
            B, model.fs_embed_att.layers[-1].out_features, device=model.device
        )

    # Fuse into final search spec embeddings
    fused_spec_embeddings = torch.cat([cls_loo_protos, att_loo_protos], dim=-1)
    fused_spec_embeddings = model.fs_spec_fuse(fused_spec_embeddings)

    # Fuse & embedding of individual exemplars for visual analysis
    if "classes" in conc_type:
        indiv_cls_embeddings = cls_embeddings
    else:
        indiv_cls_embeddings = torch.zeros(
            B, model.fs_embed_cls.layers[-1].out_features, device=model.device
        )
    if "attributes" in conc_type:
        indiv_att_embeddings = att_embeddings
    else:
        indiv_att_embeddings = torch.zeros(
            B, model.fs_embed_att.layers[-1].out_features, device=model.device
        )
    indiv_fused_spec_embeddings = torch.cat(
        [indiv_cls_embeddings, indiv_att_embeddings], dim=-1
    )
    indiv_fused_spec_embeddings = model.fs_spec_fuse(indiv_fused_spec_embeddings)

    # Setup Hungarian matcher and loss computation fn
    matcher = DeformableDetrHungarianMatcher(
        class_cost=model.detr.config.class_cost,
        bbox_cost=model.detr.config.bbox_cost,
        giou_cost=model.detr.config.giou_cost,
    )
    loss_fn = _DeformableDetrLossWithGamma(
        matcher=matcher, num_classes=1,     # Search as binary classification
        focal_alpha=0.75, losses=["labels", "boxes"]
    )
    loss_fn.focal_gamma = 2
    loss_fn = loss_fn.to(model.device)

    # Setup metric computation module
    metric_logs = MeanAveragePrecision(box_format="cxcywh")

    # Perform conditioned search for each image, with zipped encoder outputs and
    # fused search spec embeddings
    loss = 0
    for i, (enc_out, spec_emb) in enumerate(zip(detr_enc_outs, fused_spec_embeddings)):
        # Unpack encoder outputs appropriately
        enc_emb, _, spatial_shapes, _, mask_flatten = enc_out

        # Code snippet taken from DeformableDetrModel.forward(), for initializing
        # object queries and proposal bboxes from encoder outputs (to be consumed for
        # conditioned detection)
        gen_proposals_fn = model.detr.model.gen_encoder_output_proposals
        _, output_proposals = gen_proposals_fn(
            enc_emb, ~mask_flatten, spatial_shapes
        )

        # Concat object query embeddings + search spec embeddings
        embs_concat = torch.cat([
            enc_emb, spec_emb[None, None].expand_as(enc_emb)
        ], dim=-1)

        # Obtain 'search compatibility score' (as dot product btw. object queries and
        # spec embeddings) and bbox proposal for each pixel
        search_scores = torch.einsum(
            "bqd,d->bq", enc_emb, model.fs_search_match(spec_emb)
        )
        search_coord_deltas = model.fs_search_bbox(embs_concat)
        search_coord_logits = search_coord_deltas + output_proposals

        # Shape into format to be fed into loss computation fn
        search_outputs_for_loss = {
            "pred_boxes": search_coord_logits.sigmoid(),
            "logits": search_scores[..., None]
        }
        search_targets_for_loss = [
            {
                "boxes": bboxes_search_targets[i],
                "class_labels": torch.zeros(
                    len(bboxes_search_targets[i]),
                    dtype=torch.long, device=model.device
                )
            }
        ]

        # Compute and weight loss values, then aggregate to total loss
        loss_dict = loss_fn(search_outputs_for_loss, search_targets_for_loss)
        weight_dict = {
            "loss_ce": 1,
            "loss_bbox": model.detr.config.bbox_loss_coefficient,
            "loss_giou": model.detr.config.giou_loss_coefficient
        }
        loss += sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys() if k in weight_dict
        )

        # For computing metrics, it would be pointlessly wasteful to consider
        # every single pixel; let's run NMS and choose, say, top 30?
        topk = 30; iou_thres = 0.65
        topk_proposals = nms(
            search_coord_logits[0].sigmoid(), search_scores[0], iou_thres
        )

        # Compute evaluation metrics
        search_outputs_for_metric = [
            {
                "boxes": search_coord_logits[0, topk_proposals][:topk].sigmoid(),
                "scores": search_scores[0, topk_proposals][:topk],
                "labels": torch.zeros(
                    topk, dtype=torch.long, device=model.device
                )
            }
        ]
        search_targets_for_metric = [
            {
                "boxes": search_targets_for_loss[0]["boxes"],
                "labels": search_targets_for_loss[0]["class_labels"]
            }
        ]
        metric_logs.update(search_outputs_for_metric, search_targets_for_metric)

    mAPs_final = metric_logs.compute()
    metrics = {
        "mAP": mAPs_final["map"],
        "mAP@50": mAPs_final["map_50"],
        "mAP@75": mAPs_final["map_75"],
    }

    return loss, metrics, indiv_fused_spec_embeddings


class _DeformableDetrLossWithGamma(DeformableDetrLoss):
    """
    DeformableDetrLoss extended, to modify loss_label() method to enable control
    of gamma parameter for focal loss computation as well
    """
    def loss_labels(self, outputs, targets, indices, num_boxes):
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                source_logits, target_classes_onehot, num_boxes,
                alpha=self.focal_alpha, gamma=self.focal_gamma
            ) * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses
