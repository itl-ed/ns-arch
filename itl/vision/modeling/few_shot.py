"""
Helper methods factored out, which perform computations required for appropriate
exemplar-based few-shot prediction tasks (binary concept classification given
bounding boxes in image, conditioned search for bounding boxes of concept instances
in image). Methods return loss values needed for training and metric values for
evaluation.
"""
import torch
import torch.nn.functional as F

from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrHungarianMatcher,
    DeformableDetrLoss
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def compute_loss_and_metrics(
    model, pred_type, conc_type, detr_enc_outs, detr_dec_outs,
    B, N, K, bboxes_search_targets
):
    assert B == len(detr_dec_outs)
    assert N * K == B

    if pred_type == "fs_classify":
        assert bboxes_search_targets is None
        return _compute_fs_classify(
            model, conc_type, detr_dec_outs, B, N, K
        )
    else:
        assert pred_type == "fs_search"
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

    # Setup Hungarian matcher and loss computation fn
    matcher = DeformableDetrHungarianMatcher(
        class_cost=model.detr.config.class_cost,
        bbox_cost=model.detr.config.bbox_cost,
        giou_cost=model.detr.config.giou_cost,
    )
    loss_fn = DeformableDetrLoss(
        matcher=matcher, num_classes=1,     # Search as binary classification
        focal_alpha=model.detr.config.focal_alpha, losses=["labels", "boxes"]
    ).to(model.device)

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
        object_query_embeddings, output_proposals = gen_proposals_fn(
            enc_emb, ~mask_flatten, spatial_shapes
        )

        # Concat object query embeddings + search spec embeddings
        embs_concat = torch.cat([
            object_query_embeddings,
            spec_emb[None, None].expand_as(object_query_embeddings)
        ], dim=-1)

        # Obtain 'search compatibility score' (as dot product btw. object queries and
        # spec embeddings) and bbox proposal for each pixel
        search_scores = torch.einsum("bqd,d->bq", object_query_embeddings, spec_emb)
        search_coord_deltas = model.fs_search_bbox(embs_concat)
        search_coord_logits = search_coord_deltas + output_proposals

        # I think 30 would suffice for conditioned search...
        topk = 30
        topk_proposals = torch.topk(search_scores, topk, dim=1)
        topk_coords_logits = torch.gather(
            search_coord_logits, 1,
            topk_proposals.indices.unsqueeze(-1).repeat(1, 1, 4)
        )

        # Shape into format to be fed into loss computation fn
        search_outputs_for_loss = {
            "pred_boxes": topk_coords_logits.sigmoid(),
            "logits": topk_proposals.values[..., None]
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

        # Compute evaluation metrics
        search_outputs_for_metric = [
            {
                "boxes": search_outputs_for_loss["pred_boxes"][0],
                "scores": search_outputs_for_loss["logits"][0, :, 0],
                "labels": torch.zeros(
                    search_outputs_for_loss["logits"].shape[1],
                    dtype=torch.long, device=model.device
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

    return loss, metrics, fused_spec_embeddings
