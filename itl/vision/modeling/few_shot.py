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
            pos_exs_inds = set(range(i*N, i*N+K)) - {ind}
            topk = set(topk_closest_exs[ind][:K].tolist()) - {ind}
            top2k = set(topk_closest_exs[ind].tolist()) - {ind}
            r_at_k += len(topk & pos_exs_inds) / (K - 1)
            r_at_2k += len(top2k & pos_exs_inds) / (K - 1)
    metrics["recall@K"] = r_at_k / B
    metrics["recall@2K"] = r_at_2k / B

    return loss, metrics


def _compute_fs_search(
    model, conc_type, detr_enc_outs, detr_dec_outs, B, N, K, bboxes_search_targets
):
    """ (Conditioned) search mode; compute DETR loss & mAP metrics """
    assert K > 1        # Need more than one "shots" to perform search

    # Hungarian matcher
    matcher = DeformableDetrHungarianMatcher(
        class_cost=model.detr.config.class_cost,
        bbox_cost=model.detr.config.bbox_cost,
        giou_cost=model.detr.config.giou_cost,
    )
    # Loss computation fn for encoder proposal outputs
    enc_loss_fn = _DeformableDetrLossWithGamma(
        matcher=matcher, num_classes=1,     # Search as binary classification
        focal_alpha=0.75, losses=["labels"]
    )
    enc_loss_fn.focal_gamma = 2
    enc_loss_fn = enc_loss_fn.to(model.device)
    # Loss computation fn for decoder outputs
    dec_loss_fn = DeformableDetrLoss(
        matcher=matcher, num_classes=1,
        focal_alpha=0.75, losses=["labels", "boxes"]
    )
    dec_loss_fn = dec_loss_fn.to(model.device)

    # Setup metric computation module
    metric_logs = MeanAveragePrecision(box_format="cxcywh")

    loss = 0
    for i in range(N):
        for j in range(K):
            ind = i*K + j
            pos_exs_inds = list(set(range(i*N, i*N+K)) - {ind})

            # First, prime the two-stage encoder to produce proposals that are
            # generally expected to comply well with the provided search specs;
            # it's okay to take risks and include false positive here, another
            # filtering will happend at the end.

            # Compute needed concept embeddings for support instances, then average
            # prototypes to be fused into search spec embeddings
            supp_cls_embs = model.fs_embed_cls(detr_dec_outs[pos_exs_inds,0,:])
            if "classes" in conc_type:
                supp_cls_proto = supp_cls_embs.mean(dim=0)
            else:
                # No class-related condition in search spe
                supp_cls_proto = torch.zeros(
                    model.fs_embed_cls.layers[-1].out_features, device=model.device
                )
            if "attributes" in conc_type:
                supp_att_embs = model.fs_embed_att(
                    torch.cat([detr_dec_outs[pos_exs_inds,0,:], supp_cls_embs], dim=-1)
                )
                supp_att_proto = supp_att_embs.mean(dim=0)
            else:
                # No attribute-related condition in search spec
                supp_att_proto = torch.zeros(
                    model.fs_embed_att.layers[-1].out_features, device=model.device
                )

            # Fuse into search spec embedding
            fused_spec_emb = torch.cat([supp_cls_proto, supp_att_proto], dim=-1)
            fused_spec_emb = model.fs_spec_fuse(fused_spec_emb)

            # Generate (conditioned) proposals from encoder output
            enc_emb, valid_ratios, spatial_shapes, \
                level_start_index, mask_flatten = detr_enc_outs[ind]

            gen_proposals_fn = model.detr.model.gen_encoder_output_proposals
            object_query_embedding, output_proposals = gen_proposals_fn(
                enc_emb, ~mask_flatten, spatial_shapes
            )
            delta_bbox = model.detr.model.decoder.bbox_embed[-1](object_query_embedding)
            proposals_coords_logits = delta_bbox + output_proposals

            # Obtain rough 'search compatibility scores'
            embs_concat = torch.cat([
                enc_emb, fused_spec_emb[None, None].expand_as(enc_emb)
            ], dim=-1)
            search_scores = model.fs_search_match_enc(embs_concat).squeeze()

            # Compute loss for the encoder side
            loss_enc = _search_loss_and_metric(
                proposals_coords_logits[0].sigmoid(), search_scores,
                bboxes_search_targets[ind],
                model, enc_loss_fn, None
            )

            # Only keep top scoring 'config.two_stage_num_proposals' proposals
            topk = model.detr.config.two_stage_num_proposals
            topk_proposals = search_scores.topk(topk).indices
            topk_coords_logits = torch.gather(
                proposals_coords_logits, 1,
                topk_proposals[None, ..., None].expand(1, -1, 4)
            )
            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()

            # Now perform search based on the proposals from the encoder side

            # Prepare decoder inputs
            pos_trans_out = model.detr.model.get_proposal_pos_embed(topk_coords_logits)
            pos_trans_out = model.detr.model.pos_trans_norm(
                model.detr.model.pos_trans(pos_trans_out)
            )
            query_embed, target = torch.split(pos_trans_out, enc_emb.shape[-1], dim=2)

            # Pass through the decoder; code below taken & simplified from
            # DeformableDetrDecoder.forward()
            hidden_states = target
            for li, decoder_layer in enumerate(model.detr.model.decoder.layers):
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=query_embed,
                    encoder_hidden_states=enc_emb,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_attention_mask=mask_flatten
                )

                hidden_states = layer_outputs[0]

                # Iterative bounding box refinement
                tmp = model.detr.model.decoder.bbox_embed[li](hidden_states)
                new_reference_points = torch.special.logit(reference_points, eps=1e-6)
                new_reference_points = tmp + new_reference_points
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            dec_embs, last_reference_points = hidden_states, reference_points

            # Compute class/attribute-centric feature vectors for candidates as 
            # needed, then prepare fused spec embedding -- this time with weighted
            # prototypes (average feature vectors weighted by distance to support
            # instances)
            cand_cls_embs = model.fs_embed_cls(dec_embs)
            if "classes" in conc_type:
                cls_dists_sq = torch.cdist(cand_cls_embs, supp_cls_embs[None]) ** 2
                cls_exs_weights = F.softmax(-cls_dists_sq, dim=-1)[0]
                weighted_cls_protos = torch.einsum(
                    "cs,sd->cd", cls_exs_weights, supp_cls_embs
                )
            else:
                weighted_cls_protos = torch.zeros(
                    topk, model.fs_embed_cls.layers[-1].out_features,
                    device=model.device
                )
            if "attributes" in conc_type:
                cand_att_embs = model.fs_embed_att(
                    torch.cat([dec_embs, cand_cls_embs], dim=-1)
                )
                att_dists_sq = torch.cdist(cand_att_embs, supp_att_embs[None]) ** 2
                att_exs_weights = F.softmax(-att_dists_sq, dim=-1)[0]
                weighted_att_protos = torch.einsum(
                    "cs,sd->cd", att_exs_weights, supp_att_embs
                )
            else:
                weighted_att_protos = torch.zeros(
                    topk, model.fs_embed_att.layers[-1].out_features,
                    device=model.device
                )

            # Fuse weighted prototypes into search spec embeddings, adapted for each
            # candidate
            fused_adaptive_spec_embs = torch.cat(
                [weighted_cls_protos, weighted_att_protos], dim=-1
            )
            fused_adaptive_spec_embs = model.fs_spec_fuse(fused_adaptive_spec_embs)

            # Concat decoder outputs + adapted search spec embeddings for further
            # prediction
            embs_concat = torch.cat([dec_embs[0], fused_adaptive_spec_embs], dim=-1)

            # Obtain (conditioned) bbox estimates
            delta_bbox = model.fs_search_bbox(embs_concat)
            cand_coords = delta_bbox + torch.special.logit(last_reference_points[0])
            cand_coords = cand_coords.sigmoid()

            # Search compatibility scores based on support vs. candidate embeddings
            cand_scores = model.fs_search_match_dec(embs_concat).squeeze()

            # Compute loss for the decoder side, while updating stats for metrics
            loss_dec = _search_loss_and_metric(
                cand_coords, cand_scores, bboxes_search_targets[ind],
                model, dec_loss_fn, metric_logs
            )

            loss += loss_enc + loss_dec

    mAPs_final = metric_logs.compute()
    metrics = {
        "mAP": mAPs_final["map"],
        "mAP@50": mAPs_final["map_50"],
        "mAP@75": mAPs_final["map_75"],
    }

    return loss, metrics


def _search_loss_and_metric(bboxes, scores, targets, model, loss_fn, metric_module):
    # Shape into format to be fed into loss computation fn
    search_outputs_for_loss = {
        "pred_boxes": bboxes[None],
        "logits": scores[None, ..., None]
    }
    search_targets_for_loss = [
        {
            "boxes": targets,
            "class_labels": torch.zeros(
                len(targets), dtype=torch.long, device=model.device
            )
        }
    ]

    # Compute and weight loss values, then aggregate to total loss
    loss_dict = loss_fn(search_outputs_for_loss, search_targets_for_loss)
    weight_dict = {
        "loss_ce": model.detr.config.two_stage_num_proposals / len(bboxes),
        "loss_bbox": model.detr.config.bbox_loss_coefficient,
        "loss_giou": model.detr.config.giou_loss_coefficient
    }
    loss = sum(
        loss_dict[k] * weight_dict[k]
        for k in loss_dict.keys() if k in weight_dict
    )

    if metric_module is not None:
        # Compute evaluation metrics
        search_outputs_for_metric = [
            {
                "boxes": bboxes,
                "scores": scores,
                "labels": torch.zeros(
                    len(bboxes), dtype=torch.long, device=model.device
                )
            }
        ]
        search_targets_for_metric = [
            {
                "boxes": search_targets_for_loss[0]["boxes"],
                "labels": search_targets_for_loss[0]["class_labels"]
            }
        ]
        metric_module.update(search_outputs_for_metric, search_targets_for_metric)

    return loss


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
