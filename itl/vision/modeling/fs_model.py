import os
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import box_convert
from transformers import AutoFeatureExtractor, DeformableDetrForObjectDetection
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMLPPredictionHead
)


class FewShotSceneGraphGenerator(pl.LightningModule):
    """
    Few-shot visual object detection & class/attribute classification model,
    implemented by attaching lightweight MLP blocks for Deformable DETR model
    outputs. MLP blocks embed DETR output vectors onto a metric space, learned
    by few-shot training with NCA objective.
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        detr_model = self.cfg.vision.model.detr_model
        assets_dir = self.cfg.paths.assets_dir

        # Loading pre-trained deformable DETR to use as basis model
        os.environ["TORCH_HOME"] = os.path.join(assets_dir, "vision_models", "torch")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            detr_model, cache_dir=os.path.join(assets_dir, "vision_models", "detr")
        )
        self.detr = DeformableDetrForObjectDetection.from_pretrained(
            detr_model, cache_dir=os.path.join(assets_dir, "vision_models", "detr")
        )

        # New lightweight MLP heads for concept-type-specific embedding (for metric
        # learning) - not included in original deformable DETR (thus not shipped in
        # pre-trained model) and needs to be newly trained
        detr_D = self.detr.config.d_model

        # MLP heads to attach on top of decoder outputs for metric-based few-shot
        # detection (classification)
        self.fs_embed_cls = DeformableDetrMLPPredictionHead(
            input_dim=detr_D, hidden_dim=detr_D, output_dim=detr_D, num_layers=2
        )
        self.fs_embed_att = DeformableDetrMLPPredictionHead(
            input_dim=detr_D*2, hidden_dim=detr_D, output_dim=detr_D, num_layers=2
        )

        # MLP heads to attach on top of encoder outputs for metric-based few-shot
        # search (conditioned detection)
        self.fs_search_cls = DeformableDetrMLPPredictionHead(
            input_dim=detr_D, hidden_dim=detr_D, output_dim=detr_D, num_layers=2
        )
        self.fs_search_att = DeformableDetrMLPPredictionHead(
            input_dim=detr_D*2, hidden_dim=detr_D, output_dim=detr_D, num_layers=2
        )
        self.fs_search_bbox = DeformableDetrMLPPredictionHead(
            input_dim=detr_D*2, hidden_dim=detr_D, output_dim=4, num_layers=3
        )

        # Initialize self.fs_search_bbox by copying parameters from the DETR model's
        # decoder's last layer's bbox_embed (from 2nd~ layer)
        fs_bb_layers = self.fs_search_bbox.layers[1:]
        dec_last_bb_layers = self.detr.model.decoder.bbox_embed[-1].layers[1:]
        for f_l, d_l in zip(fs_bb_layers, dec_last_bb_layers):
            f_l.load_state_dict(d_l.state_dict())

        # Freeze all parameters...
        for prm in self.parameters():
            prm.requires_grad = False

        # ... except those required for training the specified task
        if self.cfg.vision.task.pred_type == "fs_classify":
            if self.cfg.vision.task.conc_type == "classes":
                # Few-shot class prediction with decoder output embeddings
                for prm in self.fs_embed_cls.parameters():
                    prm.requires_grad = True
            elif self.cfg.vision.task.conc_type == "attributes":
                # Few-shot attribute prediction with decoder output embeddings
                for prm in self.fs_embed_att.parameters():
                    prm.requires_grad = True
            else:
                raise ValueError("Invalid concept type")
        elif self.cfg.vision.task.pred_type == "fs_search":
            # Few-shot search with encoder output embeddings
            for prm in self.fs_search_cls.parameters():
                prm.requires_grad = True
            for prm in self.fs_search_att.parameters():
                prm.requires_grad = True
            for prm in self.fs_search_bbox.parameters():
                prm.requires_grad = True
        else:
            raise ValueError("Invalid task type for training")
        
        self.save_hyperparameters()

    def training_step(self, batch, _):
        loss, metrics, _, _ = self._process_batch(batch)

        self.log("train_loss", loss.item())
        for metric, val in metrics.items():
            self.log(f"train_{metric}", val)

        return loss

    def validation_step(self, batch, _):
        loss, metrics, _, _ = self._process_batch(batch)

        self.log("val_loss", loss.item(), batch_size=len(batch[0][0]))
        for metric, val in metrics.items():
            self.log(f"val_{metric}", val, batch_size=len(batch[0][0]))

    def test_step(self, batch, _):
        _, metrics, embeddings, labels = self._process_batch(batch)

        for metric, val in metrics.items():
            self.log(f"test_{metric}", val, batch_size=len(batch[0][0]))

        return embeddings, labels
    
    def test_epoch_end(self, outputs):
        """ Log instance embeddings """
        if self.logger is not None and hasattr(self.logger, "log_table"):
            embeddings, labels = tuple(zip(*outputs))
            embeddings = torch.cat(embeddings)
            labels = sum(labels, ())

            # There are typically too many vectors, not all of them need to be logged...
            # Let's downsample to K (as in config) per concept
            K = self.cfg.vision.data.num_exs_per_conc_eval
            downsampled = {
                conc: random.sample([i for i, l in enumerate(labels) if conc==l], K)
                for conc in set(labels)
            }
            data_downsampled = [
                [conc]+embeddings[ex_i].tolist()
                for conc, exs in downsampled.items() for ex_i in exs
            ]

            self.logger.log_table(
                f"embeddings_{self.cfg.vision.task.conc_type}",
                columns=["concept"] + [f"D{i}" for i in range(embeddings.shape[-1])],
                data=data_downsampled
            )

    def configure_optimizers(self):
        # Populate optimizer configs
        optim_kwargs = {}
        if "init_lr" in self.cfg.vision.optim:
            optim_kwargs["lr"] = self.cfg.vision.optim.init_lr
        if "beta1_1m" in self.cfg.vision.optim and "beta2_1m" in self.cfg.vision.optim:
            optim_kwargs["betas"] = (
                1-self.cfg.vision.optim.beta1_1m, 1-self.cfg.vision.optim.beta2_1m
            )
        if "eps" in self.cfg.vision.optim:
            optim_kwargs["eps"] = self.cfg.vision.optim.eps

        # Construct optimizer instance
        Optimizer = getattr(torch.optim, self.cfg.vision.optim.algorithm)
        optim = Optimizer(self.parameters(), **optim_kwargs)

        # Populate LR scheduler configs
        sched_kwargs = {}
        if "lr_scheduler_milestones" in self.cfg.vision.optim:
            sched_kwargs["milestones"] = [
                int(s * self.cfg.vision.optim.max_steps)
                for s in self.cfg.vision.optim.lr_scheduler_milestones
            ]
        if "lr_scheduler_gamma" in self.cfg.vision.optim:
            sched_kwargs["gamma"] = self.cfg.vision.optim.lr_scheduler_gamma

        # Construct LR scheduler instance
        Scheduler = getattr(
            torch.optim.lr_scheduler, self.cfg.vision.optim.lr_scheduler
        )
        sched = Scheduler(optim, **sched_kwargs)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step"
            }
        }

    def on_save_checkpoint(self, checkpoint):
        """
        No need to save weights for DETR; del DETR weights to leave param weights
        for the newly added prediction heads only
        """
        state_dict_filtered = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("fs_"):
                state_dict_filtered[k] = v
        checkpoint["state_dict"] = state_dict_filtered

    def _process_batch(self, batch):
        """
        Shared subroutine for processing batch to obtain loss & performance metric
        """
        self.detr.eval()        # DETR always eval mode

        batch_data, conc_labels = batch

        # Batch size, number of "ways" and "shots" in few-shot episodes
        B = len(batch_data)
        N = len(set(conc_labels))
        K = B // N

        if len(batch_data) == 3:
            # Didn't find cached pre-computed vectors, need to compute
            images, bboxes, bb_inds = batch_data
            assert len(images) == len(bboxes) == len(bb_inds)

            bboxes = [torch.tensor(bbs).to(self.device) for bbs in bboxes]
            bboxes = [box_convert(bbs, "xywh", "cxcywh") for bbs in bboxes]
            bboxes = [
                torch.stack([
                    bbs[:,0] / images[i].width, bbs[:,1] / images[i].height,
                    bbs[:,2] / images[i].width, bbs[:,3] / images[i].height,
                ], dim=-1)
                for i, bbs in enumerate(bboxes)
            ]

            # Process each image with DETR one-by-one in eval mode
            batch_fvecs = []
            for img, bbs, bbis in zip(images, bboxes, bb_inds):
                fvecs = self.fvecs_from_image_and_bboxes(img, bbs)
                batch_fvecs.append(fvecs[:,bbis])
            batch_fvecs = torch.cat(batch_fvecs)
        else:
            # Cached vectors for the batch found
            batch_fvecs = batch_data.to(self.device)

        # Pass through MLP block according to specified task concept type
        if self.cfg.vision.task.conc_type == "classes":
            conc_embedding = self.fs_embed_cls(batch_fvecs[:,0,:])
        elif self.cfg.vision.task.conc_type == "attributes":
            cls_embedding = self.fs_embed_cls(batch_fvecs[:,0,:])
            conc_embedding = self.fs_embed_att(
                torch.cat([batch_fvecs[:,0,:], cls_embedding], dim=-1)
            )
        else:
            raise ValueError("Invalid concept type")

        # Obtain pairwise distances (squared) between exemplars
        pairwise_dists_sq = torch.cdist(conc_embedding, conc_embedding, p=2.0) ** 2
        pairwise_dists_sq.fill_diagonal_(float("inf"))     # Disregarding reflexive distance

        # Compute NCA loss
        pos_dist_mask = torch.zeros_like(pairwise_dists_sq)
        for i in range(N):
            pos_dist_mask[i*K:(i+1)*K, i*K:(i+1)*K] = 1.0
        pos_dist_mask.fill_diagonal_(0.0)
        loss_nca = F.softmax(-pairwise_dists_sq) * pos_dist_mask
        loss_nca = -loss_nca.sum(dim=-1).log().sum() / B

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

        return loss_nca, metrics, conc_embedding, conc_labels

    def fvecs_from_image_and_bboxes(self, image, bboxes):
        """
        Subroutine for extracting feature vectors corresponding to image-bbox
        pairs; code below is composed from snippets taken from huggingface's
        original Deformable DETR code components, appropriately abridging unused
        parts and making appropriate modifications to accommodate our needs
        """
        encoder_outputs, valid_ratios, spatial_shapes, \
            level_start_index, mask_flatten = self._detr_enc_outputs(image)
        decoder_outputs = self._detr_dec_outputs(
            encoder_outputs, bboxes,
            valid_ratios, spatial_shapes, level_start_index, mask_flatten
        )

        return decoder_outputs

    def _detr_enc_outputs(self, image):
        """ Subroutine for processing images up to encoder outputs """
        detr_cfg = self.detr.config

        # Input preprocessing
        inputs = self.feature_extractor(
            images=image, return_tensors="pt"
        )
        inputs = { k: v.to(self.device) for k, v in inputs.items() }

        # Code below taken & simplified from DeformableDetrModel.forward()
        pixel_values = inputs["pixel_values"]
        pixel_mask = inputs["pixel_mask"]

        # Extract multi-scale feature maps of same resolution `config.d_model`
        # (cf Figure 4 in paper). First, send pixel_values + pixel_mask through
        # backbone to obtain the features which is a list of tuples
        features, position_embeddings_list = self.detr.model.backbone(
            pixel_values, pixel_mask
        )

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(self.detr.model.input_proj[level](source))
            masks.append(mask)

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions
        # on the final stage
        if detr_cfg.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, detr_cfg.num_feature_levels):
                if level == _len_sources:
                    source = self.detr.model.input_proj[level](features[-1][0])
                else:
                    source = self.detr.model.input_proj[level](sources[-1])
                mask = F.interpolate(pixel_mask[None].float(), size=source.shape[-2:])
                mask = mask.to(torch.bool)[0]
                pos_l = self.detr.model.backbone.position_embedding(source, mask)
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
            lvl_pos_embed = pos_embed + self.detr.model.level_embed[level].view(1, 1, -1)
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
        valid_ratios = torch.stack([self.detr.model.get_valid_ratio(m) for m in masks], 1)

        # revert valid_ratios
        valid_ratios = ~valid_ratios.bool()
        valid_ratios = valid_ratios.float()

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten
        # (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        encoder_outputs = self.detr.model.encoder(
            inputs_embeds=source_flatten,
            attention_mask=mask_flatten,
            position_embeddings=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios
        )

        return encoder_outputs, valid_ratios, spatial_shapes, level_start_index, mask_flatten

    def _detr_dec_outputs(
        self, enc_out, bboxes,
        valid_ratios, spatial_shapes, level_start_index, mask_flatten
    ):
        """ Subroutine for processing encoder outputs to obtain decoder outputs """
        detr_cfg = self.detr.config

        # Fifth, prepare decoder inputs. In addition to the top-k regional proposals
        # for two-stage DETR, prepare query embedding(s) corresponding to the provided
        # bbox(es).
        num_channels = enc_out[0].shape[-1]
        object_query_embedding, output_proposals = self.detr.model.gen_encoder_output_proposals(
            enc_out[0], ~mask_flatten, spatial_shapes
        )

        # apply a detection head to each pixel (A.4 in paper)
        # linear projection for bounding box binary classification (i.e. foreground and background)
        enc_outputs_class = self.detr.model.decoder.class_embed[-1](object_query_embedding)
        # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
        delta_bbox = self.detr.model.decoder.bbox_embed[-1](object_query_embedding)
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
        pos_trans_out = self.detr.model.get_proposal_pos_embed(reference_points_logits)
        pos_trans_out = self.detr.model.pos_trans_norm(
            self.detr.model.pos_trans(pos_trans_out)
        )
        query_embed, target = torch.split(pos_trans_out, num_channels, dim=2)

        # Feed prepared inputs to decoder; code below taken & simplified from
        # DeformableDetrDecoder.forward()
        hidden_states = target
        for i, decoder_layer in enumerate(self.detr.model.decoder.layers):
            reference_points_input = reference_points[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=query_embed,
                encoder_hidden_states=enc_out[0],
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                encoder_attention_mask=mask_flatten
            )

            hidden_states = layer_outputs[0]

            # Iterative bounding box refinement, except for proposals with bboxes provided
            tmp = self.detr.model.decoder.bbox_embed[i](hidden_states)
            new_reference_points = torch.special.logit(reference_points, eps=1e-6)
            new_reference_points[:,bboxes.shape[0]:] = \
                tmp[:,bboxes.shape[0]:] + new_reference_points[:,bboxes.shape[0]:]
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

        # Return parts of final decoder layer output corresponding to the provided
        # bboxes
        return hidden_states[:,:bboxes.shape[0]]
