"""
Helper methods factored out, which are essentially abridgement of the original
(i.e. Huggingface's) implementation of Deformable DETR, specifically computation
of encoder & decoder outputs
"""
import torch
import torch.nn.functional as F


def detr_enc_outputs(detr_model, image, feature_extractor):
    """ Subroutine for processing images up to encoder outputs """
    detr_cfg = detr_model.config

    # Input preprocessing
    inputs = feature_extractor(
        images=image, return_tensors="pt"
    )
    inputs = { k: v.to(detr_model.device) for k, v in inputs.items() }

    # Code below taken & simplified from DeformableDetrModel.forward()
    pixel_values = inputs["pixel_values"]
    pixel_mask = inputs["pixel_mask"]

    # Extract multi-scale feature maps of same resolution `config.d_model`
    # (cf Figure 4 in paper). First, send pixel_values + pixel_mask through
    # backbone to obtain the features which is a list of tuples
    features, position_embeddings_list = detr_model.model.backbone(
        pixel_values, pixel_mask
    )

    # Then, apply 1x1 convolution to reduce the channel dimension to d_model
    sources = []
    masks = []
    for level, (source, mask) in enumerate(features):
        sources.append(detr_model.model.input_proj[level](source))
        masks.append(mask)

    # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions
    # on the final stage
    if detr_cfg.num_feature_levels > len(sources):
        _len_sources = len(sources)
        for level in range(_len_sources, detr_cfg.num_feature_levels):
            if level == _len_sources:
                source = detr_model.model.input_proj[level](features[-1][0])
            else:
                source = detr_model.model.input_proj[level](sources[-1])
            mask = F.interpolate(pixel_mask[None].float(), size=source.shape[-2:])
            mask = mask.to(torch.bool)[0]
            pos_l = detr_model.model.backbone.position_embedding(source, mask)
            pos_l = pos_l.to(source.dtype)
            sources.append(source)
            masks.append(mask)
            position_embeddings_list.append(pos_l)
    
    # Prepare encoder inputs (by flattening)
    source_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    zipped = enumerate(zip(sources, masks, position_embeddings_list))
    for level, (source, mask, pos_embed) in zipped:
        _, _, height, width = source.shape
        spatial_shape = (height, width)
        spatial_shapes.append(spatial_shape)
        source = source.flatten(2).transpose(1, 2)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed = pos_embed + detr_model.model.level_embed[level].view(1, 1, -1)
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        source_flatten.append(source)
        mask_flatten.append(mask)
    source_flatten = torch.cat(source_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=source_flatten.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)),
        spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    valid_ratios = torch.stack([detr_model.model.get_valid_ratio(m) for m in masks], 1)

    # revert valid_ratios
    valid_ratios = ~valid_ratios.bool()
    valid_ratios = valid_ratios.float()

    # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten
    # (backbone + proj layer output) through encoder
    # Also provide spatial_shapes, level_start_index and valid_ratios
    encoder_outputs = detr_model.model.encoder(
        inputs_embeds=source_flatten,
        attention_mask=mask_flatten,
        position_embeddings=lvl_pos_embed_flatten,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios
    )

    return encoder_outputs[0], valid_ratios, spatial_shapes, level_start_index, mask_flatten

def detr_dec_outputs(
    detr_model, enc_out, bboxes, lock_provided_boxes,
    valid_ratios, spatial_shapes, level_start_index, mask_flatten
):
    """ Subroutine for processing encoder outputs to obtain decoder outputs """
    detr_cfg = detr_model.config

    # Fifth, prepare decoder inputs. In addition to the top-k regional proposals
    # for two-stage DETR, prepare query embedding(s) corresponding to the provided
    # bbox(es).
    num_channels = enc_out.shape[-1]
    object_query_embedding, output_proposals = detr_model.model.gen_encoder_output_proposals(
        enc_out, ~mask_flatten, spatial_shapes
    )

    # apply a detection head to each pixel (A.4 in paper)
    # linear projection for bounding box binary classification (i.e. foreground and background)
    enc_outputs_class = detr_model.model.decoder.class_embed[-1](object_query_embedding)
    # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
    delta_bbox = detr_model.model.decoder.bbox_embed[-1](object_query_embedding)
    enc_outputs_coord_logits = delta_bbox + output_proposals

    # only keep top scoring `config.two_stage_num_proposals` proposals
    topk = detr_cfg.two_stage_num_proposals
    topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
    topk_coords_logits = torch.gather(
        enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
    )

    topk_coords_logits = topk_coords_logits.detach()
    reference_points = topk_coords_logits.sigmoid()

    # Add proposals from provided bboxes
    reference_points = torch.cat([bboxes[None], reference_points], dim=1)
    reference_points_logits = torch.cat([
        torch.special.logit(bboxes[None], eps=1e-6), topk_coords_logits
    ], dim=1)

    pos_trans_out = detr_model.model.get_proposal_pos_embed(reference_points_logits)
    pos_trans_out = detr_model.model.pos_trans_norm(
        detr_model.model.pos_trans(pos_trans_out)
    )
    query_embed, target = torch.split(pos_trans_out, num_channels, dim=2)

    # Feed prepared inputs to decoder; code below taken & simplified from
    # DeformableDetrDecoder.forward()
    hidden_states = target
    for i, decoder_layer in enumerate(detr_model.model.decoder.layers):
        reference_points_input = reference_points[:, :, None] * \
            torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        
        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings=query_embed,
            encoder_hidden_states=enc_out,
            reference_points=reference_points_input,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            encoder_attention_mask=mask_flatten
        )

        hidden_states = layer_outputs[0]

        # Iterative bounding box refinement
        tmp = detr_model.model.decoder.bbox_embed[i](hidden_states)
        new_reference_points = torch.special.logit(reference_points, eps=1e-6)
        if lock_provided_boxes:
            # ... except for proposals with bboxes provided
            new_reference_points[:,bboxes.shape[0]:] = \
                tmp[:,bboxes.shape[0]:] + new_reference_points[:,bboxes.shape[0]:]
        else:
            # ... for all
            new_reference_points = tmp + new_reference_points
        new_reference_points = new_reference_points.sigmoid()
        reference_points = new_reference_points.detach()

    # Return parts of final decoder layer output corresponding to the provided
    # bboxes, and last layer's reference point output (needed for final bounding
    # box computation in ensemble prediction)
    return hidden_states, reference_points
