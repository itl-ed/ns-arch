"""
Helper methods factored out, which perform computations required for appropriate
exemplar-based few-shot prediction tasks (binary concept classification given
bounding boxes in image, conditioned search for bounding boxes of concept instances
in image). Methods return loss values needed for training and metric values for
evaluation.
"""
import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrHungarianMatcher,
    DeformableDetrLoss,
    generalized_box_iou,
    center_to_corners_format
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def compute_fs_classify(model, conc_type, detr_dec_outs, N, K):
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
    
    B = N * K
    metrics["recall@K"] = r_at_k / B
    metrics["recall@2K"] = r_at_2k / B

    return loss, metrics


def compute_fs_search(
    model, conc_type, detr_enc_outs, bboxes, bb_inds, supp_vecs
):
    """ (Conditioned) search mode; compute DETR loss & mAP metrics """
    # Hungarian matchers
    matcher_enc = DeformableDetrHungarianMatcher(
        class_cost=model.detr.config.class_cost,
        bbox_cost=model.detr.config.bbox_cost,
        giou_cost=model.detr.config.giou_cost,
    )
    matcher_dec = _HungarianMatcherWithPerConceptBoxes(
        class_cost=model.detr.config.class_cost,
        bbox_cost=model.detr.config.bbox_cost,
        giou_cost=model.detr.config.giou_cost,
    )

    # Setup metric computation module
    metric_logs = MeanAveragePrecision(box_format="cxcywh")

    loss = 0
    for enc_out, bbs, inds, vecs in zip(detr_enc_outs, bboxes, bb_inds, supp_vecs):
        conds_lists = [[]] * len(inds)
        supp_cls_embs = [model.fs_embed_cls(vs[:,0,:]) for vs in vecs]
        if "classes" in conc_type:
            conds_lists = [
                cnds + [(("cls"), ces)]
                for cnds, ces in zip(conds_lists, supp_cls_embs)
            ]
        if "attributes" in conc_type:
            supp_att_embs = [
                model.fs_embed_att(torch.cat([vs[:,0,:], ces], dim=-1))
                for vs, ces in zip(vecs, supp_cls_embs)
            ]
            conds_lists = [
                cnds + [(("att"), aes)]
                for cnds, aes in zip(conds_lists, supp_att_embs)
            ]
        
        proposals_coords, enc_search_scores, outputs_coords, dec_search_scores \
            = few_shot_search_img(model, enc_out, conds_lists)

        # Ground-truth target to predict (Wrap once in a list to pass as a 'batch'
        # with a single entry)
        target = [
            {
                "boxes": torch.stack([
                    bbs[bi] for per_conc in inds for bi in per_conc
                ], dim=0),
                "class_labels": torch.tensor([
                    ci for ci, per_conc in enumerate(inds) for _ in per_conc
                ], dtype=torch.long, device=model.device)
            }
        ]

        # Compute loss for the encoder side
        # Loss computation fn for encoder proposal outputs
        enc_loss_fn = DeformableDetrLoss(
            matcher=matcher_enc, num_classes=len(inds),
            focal_alpha=0.25, losses=["labels", "boxes"]
        )
        enc_loss_fn = enc_loss_fn.to(model.device)
        preds_enc = {
            "pred_boxes": proposals_coords[None],
            "logits": enc_search_scores[None]
        }
        loss_enc = _search_loss_and_metric(
            preds_enc, target, model, enc_loss_fn, None
        )

        # Compute loss for the decoder side, while updating stats for metrics
        # Loss computation fn for decoder outputs
        dec_loss_fn = _DeformableDetrLossWithPerConceptBoxes(
            matcher=matcher_dec, num_classes=len(inds),
            focal_alpha=0.25, losses=["labels", "boxes"]
        )
        dec_loss_fn = dec_loss_fn.to(model.device)
        preds_dec = {
            "pred_boxes": outputs_coords[None],
            "logits": dec_search_scores[None]
        }
        loss_dec = _search_loss_and_metric(
            preds_dec, target, model, dec_loss_fn, metric_logs
        )

        loss += loss_enc + loss_dec

    mAPs_final = metric_logs.compute()
    metrics = {
        "mAP": mAPs_final["map"],
        "mAP@50": mAPs_final["map_50"],
        "mAP@75": mAPs_final["map_75"],
    }

    return loss, metrics


def _search_loss_and_metric(preds, targets, model, loss_fn, metric_module):
    # Compute and weight loss values, then aggregate to total loss
    loss_dict = loss_fn(preds, targets)
    weight_dict = {
        "loss_ce": model.detr.config.two_stage_num_proposals / preds["logits"].shape[1],
        "loss_bbox": model.detr.config.bbox_loss_coefficient,
        "loss_giou": model.detr.config.giou_loss_coefficient
    }
    loss = sum(
        loss_dict[k] * weight_dict[k]
        for k in loss_dict.keys() if k in weight_dict
    )

    if metric_module is not None:
        # Compute evaluation metrics
        max_classes = preds["logits"][0].max(dim=-1)
        preds_for_metric = [
            {
                "boxes": torch.stack([
                    bbs[ci]
                    for bbs, ci in zip(preds["pred_boxes"][0], max_classes.indices)
                ]),
                "scores": max_classes.values,
                "labels": max_classes.indices
            }
        ]
        targets_for_metric = [
            {
                "boxes": targets[0]["boxes"],
                "labels": targets[0]["class_labels"]
            }
        ]
        metric_module.update(preds_for_metric, targets_for_metric)

    return loss


def few_shot_search_img(model, enc_out, conds_lists):
    # Number of concepts being searched; roughly equivalent to "way" in few-shot
    # learning episodes
    N = len(conds_lists)

    # First, prime the two-stage encoder to produce proposals that are
    # generally expected to comply well with the provided search specs;
    # it's okay to take risks and include false positive here, another
    # filtering will happend at the end.

    # Compute needed concept embeddings for support instances, then average
    # prototypes to be fused into search spec embeddings
    cls_proto_sum = torch.zeros(N, model.detr.config.d_model, device=model.device)
    att_proto_sum = torch.zeros(N, model.detr.config.d_model, device=model.device)
    for i, cnds in enumerate(conds_lists):
        for conc_type, pos_exs_vecs in cnds:
            pos_proto = torch.tensor(pos_exs_vecs).to(model.device).mean(dim=0)
            if conc_type == "cls":
                cls_proto_sum[i] = cls_proto_sum[i] + pos_proto
            elif conc_type == "att":
                att_proto_sum[i] = att_proto_sum[i] + pos_proto
            else:
                # Dunno what should happen here yet...
                raise NotImplementedError

    # Fuse into search spec embedding
    fused_spec_emb = torch.cat([cls_proto_sum, att_proto_sum], dim=-1)
    fused_spec_emb = model.fs_spec_fuse(fused_spec_emb)

    # Generate (conditioned) proposals from encoder output
    enc_emb, valid_ratios, spatial_shapes, \
            level_start_index, mask_flatten = enc_out

    gen_proposals_fn = model.detr.model.gen_encoder_output_proposals
    object_query_embedding, output_proposals = gen_proposals_fn(
        enc_emb, ~mask_flatten, spatial_shapes
    )
    delta_bbox = model.detr.model.decoder.bbox_embed[-1](object_query_embedding)
    proposals_coords_logits = delta_bbox + output_proposals

    # Obtain rough 'search compatibility scores'
    num_pixels = enc_emb.shape[1]
    embs_concat = torch.cat([
        enc_emb[0,:,None,:].expand(-1, N, -1),
        fused_spec_emb[None,:,:].expand(num_pixels, -1, -1)
    ], dim=-1)
    rough_scores = model.fs_search_match_enc(embs_concat)[..., 0]

    # Only keep top scoring 'config.two_stage_num_proposals' proposals
    topk = model.detr.config.two_stage_num_proposals
    topk_proposals = rough_scores.max(dim=-1).values.topk(topk).indices
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

    cand_cls_embs = model.fs_embed_cls(dec_embs)
    cand_att_embs = model.fs_embed_att(torch.cat([dec_embs, cand_cls_embs], dim=-1))
    cand_embs = { "cls": cand_cls_embs, "att": cand_att_embs }

    # Compute class/attribute-centric feature vectors for candidates as 
    # needed, then prepare fused spec embedding -- this time with weighted
    # prototypes (average feature vectors weighted by distance to support
    # instances)
    cls_w_proto_sums = torch.zeros(
        dec_embs.shape[1], N, model.detr.config.d_model, device=model.device
    )
    att_w_proto_sums = torch.zeros(
        dec_embs.shape[1], N, model.detr.config.d_model, device=model.device
    )
    for i, cnds in enumerate(conds_lists):
        for conc_type, pos_exs_vecs in cnds:
            if not isinstance(pos_exs_vecs, torch.Tensor):
                pos_exs_vecs = torch.tensor(pos_exs_vecs, device=model.device)

            dists_sq = torch.cdist(cand_embs[conc_type], pos_exs_vecs[None]) ** 2
            exs_weights = F.softmax(-dists_sq[0], dim=-1)
            pos_w_protos = torch.einsum(
                "cs,sd->cd", exs_weights, pos_exs_vecs
            )

            if conc_type == "cls":
                cls_w_proto_sums[:,i,:] = cls_w_proto_sums[:,i,:] + pos_w_protos
            elif conc_type == "att":
                att_w_proto_sums[:,i,:] = att_w_proto_sums[:,i,:] + pos_w_protos
            else:
                # Dunno what should happen here yet...
                raise NotImplementedError

    # Fuse weighted prototypes into search spec embeddings, adapted for each
    # candidate
    fused_adaptive_spec_embs = torch.cat(
        [cls_w_proto_sums, att_w_proto_sums], dim=-1
    )
    fused_adaptive_spec_embs = model.fs_spec_fuse(fused_adaptive_spec_embs)

    # Concat decoder outputs + adapted search spec embeddings for further
    # prediction
    embs_concat = torch.cat([
        dec_embs[0,:,None,:].expand(-1, N, -1), fused_adaptive_spec_embs
    ], dim=-1)

    # Obtain (conditioned) bbox estimates
    delta_bbox = model.fs_search_bbox(embs_concat)
    cand_coords_logits = delta_bbox + \
        torch.special.logit(last_reference_points[0,:,None,:])

    # Search compatibility scores based on support vs. candidate embeddings
    cand_scores = model.fs_search_match_dec(embs_concat)[..., 0]

    # Output values: Rough encoder-side proposal boxes and scores + final decoder-side
    # prediction boxes and scores
    return (
        proposals_coords_logits[0].sigmoid(), rough_scores,
        cand_coords_logits.sigmoid(), cand_scores
    )


class _HungarianMatcherWithPerConceptBoxes(DeformableDetrHungarianMatcher):
    """
    DeformableDetrLoss extended, to modify forward() to handle concept-sensitive
    box predictions 
    """
    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, num_classes, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes - concept-sensitive version
        bbox_cost = torch.cat([
            torch.cdist(out_bbox[:,i,:], bb[None], p=1)
            for i, bb in zip(target_ids, target_bbox)
        ], dim=-1)

        # Compute the giou cost between boxes - concept-sensitive version
        giou_cost = torch.cat([
            -generalized_box_iou(
                center_to_corners_format(out_bbox[:,i,:]),
                center_to_corners_format(bb[None])
            )
            for i, bb in zip(target_ids, target_bbox)
        ], dim=-1)

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class _DeformableDetrLossWithPerConceptBoxes(DeformableDetrLoss):
    """
    DeformableDetrLoss extended, to modify loss_boxes() to handle concept-sensitive
    box predictions 
    """
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Also need target id info for concept-sensitive box choice & loss computation
        target_ids = torch.cat([t["class_labels"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = torch.stack([
            F.l1_loss(s_bb[i], t_bb, reduction="none")
            for s_bb, i, t_bb in zip(source_boxes, target_ids, target_boxes)
        ])

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = torch.stack([
            1 - torch.diag(
                generalized_box_iou(
                    center_to_corners_format(s_bb[i, None]),
                    center_to_corners_format(t_bb[None])
                )
            )
            for s_bb, i, t_bb in zip(source_boxes, target_ids, target_boxes)
        ])
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses
