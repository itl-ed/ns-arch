import os
import random
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.ops import box_convert, clip_boxes_to_image
from transformers import AutoFeatureExtractor, DeformableDetrForObjectDetection
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMLPPredictionHead
)

from .detr_abridged import detr_enc_outputs, detr_dec_outputs
from .few_shot import compute_fs_classify, compute_fs_search, few_shot_search_img


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

        # MLP heads to attach on top of encoder & decoder outputs for metric-based
        # few-shot search (conditioned detection)
        self.fs_spec_fuse = DeformableDetrMLPPredictionHead(
            input_dim=detr_D*2, hidden_dim=detr_D, output_dim=detr_D, num_layers=2
        )
        self.fs_search_match_enc = DeformableDetrMLPPredictionHead(
            input_dim=detr_D*2, hidden_dim=detr_D, output_dim=1, num_layers=2
        )
        self.fs_search_match_dec = DeformableDetrMLPPredictionHead(
            input_dim=detr_D*2, hidden_dim=detr_D, output_dim=1, num_layers=2
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
        if "task" in self.cfg.vision:
            if self.cfg.vision.task == "fs_classify":
                # Few-shot concept classification with decoder output embeddings
                for prm in self.fs_embed_cls.parameters():
                    prm.requires_grad = True
                for prm in self.fs_embed_att.parameters():
                    prm.requires_grad = True
            elif self.cfg.vision.task == "fs_search":
                # Few-shot search with encoder/decoder output embeddings
                for prm in self.fs_spec_fuse.parameters():
                    prm.requires_grad = True
                for prm in self.fs_search_match_enc.parameters():
                    prm.requires_grad = True
                for prm in self.fs_search_match_dec.parameters():
                    prm.requires_grad = True
                for prm in self.fs_search_bbox.parameters():
                    prm.requires_grad = True
            else:
                raise ValueError("Invalid task type for training")
        
        self.save_hyperparameters()

    def training_step(self, batch, *_):
        if isinstance(batch, tuple):
            # Simpler case of single batch from single dataloader
            loss, metrics = self._process_batch(batch)

            self.log("train_loss", loss.item())
            for metric, val in metrics.items():
                self.log(f"train_{metric}", val)

            return loss
        else:
            # 'Batch' consists of batches from multiple dataloaders (most likely
            # in fs_search task mode); process and log each
            loss_total = 0
            for b in batch:
                loss, metrics = self._process_batch(b)
                loss_total += loss

                conc_type = b[1]
                if isinstance(conc_type, tuple):
                    conc_type = "+".join(conc_type)

                self.log(f"train_loss_{conc_type}", loss.item())
                for metric, val in metrics.items():
                    self.log(f"train_{metric}_{conc_type}", val)

            self.log(f"train_loss", loss_total.item())

            return loss_total

    def validation_step(self, batch, *_):
        loss, metrics = self._process_batch(batch)
        return loss, metrics, batch[1]

    def validation_epoch_end(self, outputs):
        if len(self.trainer.val_dataloaders) == 1:
            outputs = [outputs]

        avg_losses = []
        for outputs_per_dataloader in outputs:
            if len(outputs_per_dataloader) == 0:
                continue

            conc_type = outputs_per_dataloader[0][2]
            if isinstance(conc_type, tuple):
                conc_type = "+".join(conc_type)

            # Log epoch average loss
            avg_loss = torch.stack([
                loss for loss, _, _ in outputs_per_dataloader
            ])
            avg_loss = avg_loss.mean()
            self.log(
                f"val_loss_{conc_type}", avg_loss.item(), add_dataloader_idx=False
            )
            avg_losses.append(avg_loss.item())

            # Log epoch average metrics
            avg_metrics = defaultdict(list)
            for metric_type in outputs_per_dataloader[0][1]:
                for _, metrics, _ in outputs_per_dataloader:
                    avg_metrics[metric_type].append(metrics[metric_type])
            
            for metric_type, vals in avg_metrics.items():
                avg_val = sum(vals) / len(vals)
                self.log(
                    f"val_{metric_type}_{conc_type}", avg_val, add_dataloader_idx=False
                )

        # Total validation loss
        final_avg_loss = sum(avg_losses) / len(avg_losses)
        self.log(f"val_loss", final_avg_loss, add_dataloader_idx=False)

    def test_step(self, batch, *_):
        _, metrics = self._process_batch(batch)
        return metrics, batch[1]
    
    def test_epoch_end(self, outputs):
        if len(self.trainer.test_dataloaders) == 1:
            outputs = [outputs]

        for outputs_per_dataloader in outputs:
            if len(outputs_per_dataloader) == 0:
                continue

            conc_type = outputs_per_dataloader[0][1]
            if isinstance(conc_type, tuple):
                conc_type = "+".join(conc_type)

            # Log epoch average metrics
            avg_metrics = defaultdict(list)
            for metric_type in outputs_per_dataloader[0][0]:
                for metrics, _ in outputs_per_dataloader:
                    avg_metrics[metric_type].append(metrics[metric_type])
            
            for metric_type, vals in avg_metrics.items():
                avg_val = sum(vals) / len(vals)
                self.log(
                    f"test_{metric_type}_{conc_type}", avg_val, add_dataloader_idx=False
                )

    def configure_optimizers(self):
        # Populate optimizer configs
        optim_kwargs = {}
        if "init_lr" in self.cfg.vision.optim:
            optim_kwargs["lr"] = self.cfg.vision.optim.init_lr
        if self.cfg.vision.optim.algorithm == "SGD":
            if "momentum_1m" in self.cfg.vision.optim:
                optim_kwargs["momentum"] = 1-self.cfg.vision.optim.momentum_1m
        if self.cfg.vision.optim.algorithm == "Adam":
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

    def forward(self, image, bboxes=None, lock_provided_boxes=True):
        """
        Purposed as the most general endpoint of the vision module for inference,
        which takes an image as input and returns 'raw output' consisting of the
        following types of data:
            1) a class-centric embedding (projection from DETR decoder output)
            2) an attribute-centric embedding (projection from DETR decoder output)
            3) a bounding box (from DETR decoder)
        for each object (candidate) detected.
        
        Can be optionally provided with a list of additional bounding boxes as
        references to regions which are guaranteed to enclose an object each --
        so that their concept identities can be classified.
        """
        # Corner- to center-format, then absolute to relative bbox dimensions
        if bboxes is not None:
            bboxes = box_convert(bboxes, "xywh", "cxcywh")
            bboxes = torch.stack([
                bboxes[:,0] / image.width, bboxes[:,1] / image.height,
                bboxes[:,2] / image.width, bboxes[:,3] / image.height,
            ], dim=-1)
        else:
            bboxes = torch.tensor([]).view(0, 4).to(self.device)

        encoder_outputs_all = detr_enc_outputs(
            self.detr, image, self.feature_extractor
        )
        encoder_outputs, valid_ratios, spatial_shapes, \
            level_start_index, mask_flatten = encoder_outputs_all

        decoder_outputs, last_reference_points = detr_dec_outputs(
            self.detr, encoder_outputs, bboxes, lock_provided_boxes,
            valid_ratios, spatial_shapes, level_start_index, mask_flatten
        )

        # Class/attribute-centric feature vectors
        cls_embeddings = self.fs_embed_cls(decoder_outputs[0])
        att_embeddings = self.fs_embed_att(
            torch.cat([decoder_outputs[0], cls_embeddings], dim=-1)
        )

        # Obtain final bbox estimates
        dec_last_layer_ind = self.detr.config.decoder_layers
        last_bbox_embed = self.detr.bbox_embed[dec_last_layer_ind-1]
        delta_bbox = last_bbox_embed(decoder_outputs[0])

        final_bboxes = delta_bbox + torch.special.logit(last_reference_points[0])
        final_bboxes = final_bboxes.sigmoid()
        final_bboxes = torch.cat([bboxes, final_bboxes[bboxes.shape[0]:]])

        # Relative to absolute bbox dimensions, then center- to corner-format
        final_bboxes = torch.stack([
            final_bboxes[:,0] * image.width, final_bboxes[:,1] * image.height,
            final_bboxes[:,2] * image.width, final_bboxes[:,3] * image.height,
        ], dim=-1)
        final_bboxes = box_convert(final_bboxes, "cxcywh", "xywh")

        return cls_embeddings, att_embeddings, final_bboxes

    def search(self, image, conds_lists, k=None):
        """
        For few-shot search (exemplar-based conditioned detection). Given an image
        and a list of (concept type, exemplar vector set) pairs, find and return
        the top k **candidate** region proposals. The proposals do not intend to be
        highly accurate at this stage, as they will be further processed by the full
        model and tested again.
        """
        enc_out = detr_enc_outputs(self.detr, image, self.feature_extractor)
        _, _, outputs_coords, outputs_scores = \
            few_shot_search_img(self, enc_out, conds_lists)

        # Relative to absolute bbox dimensions, then center- to corner-format
        outputs_coords = torch.stack([
            outputs_coords[:,:,0] * image.width, outputs_coords[:,:,1] * image.height,
            outputs_coords[:,:,2] * image.width, outputs_coords[:,:,3] * image.height
        ], dim=-1)
        outputs_coords = box_convert(outputs_coords, "cxcywh", "xyxy")
        outputs_coords = clip_boxes_to_image(
            outputs_coords, (image.height, image.width)
        )

        if k is None:
            k = len(outputs_coords)
        topk_inds = outputs_scores.max(dim=-1).values.topk(k).indices

        return outputs_coords[topk_inds], outputs_scores[topk_inds]

    def _process_batch(self, batch):
        """
        Shared subroutine for processing batch to obtain loss & performance metric
        """
        self.detr.eval()        # DETR always eval mode

        batch_data, conc_type = batch

        if "image" in batch_data:
            images = batch_data["image"]
            bboxes = batch_data["bboxes"]

            bboxes = [box_convert(bbs, "xywh", "cxcywh") for bbs in bboxes]
            bboxes = [
                torch.stack([
                    bbs[:,0] / images[i].width, bbs[:,1] / images[i].height,
                    bbs[:,2] / images[i].width, bbs[:,3] / images[i].height,
                ], dim=-1)
                for i, bbs in enumerate(bboxes)
            ]

            # Batch size, number of "ways" and "shots" in few-shot episodes
            B = len(images)
        else:
            assert "dec_out_cached" in batch_data
            B = len(batch_data["dec_out_cached"])

        if self.cfg.vision.task == "fs_classify":
            conc_labels = batch_data["concept_label"]
            N = len(set(conc_labels))
            K = B // N

            # DETR Decoder output vectors, either computed from scratch or cached
            # values retrieved
            detr_dec_outs = []
            if "dec_out_cached" in batch_data:
                # Decoder outputs are pre-computed and retrieved
                detr_dec_outs = batch_data["dec_out_cached"]
            else:
                # Need to compute from images in batch

                # Process each image with DETR - one-by-one
                for img, bbs, bbis in zip(images, bboxes, zip(*batch_data["bb_inds"])):
                    bbis = torch.stack(bbis)
                    enc_out, dec_out = self.fvecs_from_image_and_bboxes(img, bbs)
                    detr_dec_outs.append(dec_out[0,bbis])

                detr_dec_outs = torch.stack(detr_dec_outs, dim=0)
            
            return compute_fs_classify(self, conc_type, detr_dec_outs, N, K)

        else:
            detr_enc_outs = []
            # Few-shot search task needs encoder outputs (per-pixel vectors)
            assert images is not None
            for img in images:
                enc_out = detr_enc_outputs(
                    self.detr, img, self.feature_extractor
                )
                detr_enc_outs.append(enc_out)

            return compute_fs_search(
                self, conc_type, detr_enc_outs, bboxes,
                batch_data["bb_inds"], batch_data["supp_vecs"]
            )

    def fvecs_from_image_and_bboxes(self, image, bboxes):
        """
        Subroutine for extracting feature vectors corresponding to image-bbox
        pairs; code below is composed from snippets taken from huggingface's
        original Deformable DETR code components, appropriately abridging unused
        parts and making appropriate modifications to accommodate our needs
        """
        encoder_outputs_all = detr_enc_outputs(
            self.detr, image, self.feature_extractor
        )
        encoder_outputs, valid_ratios, spatial_shapes, \
            level_start_index, mask_flatten = encoder_outputs_all

        decoder_outputs, _ = detr_dec_outputs(
            self.detr, encoder_outputs, bboxes, True,
            valid_ratios, spatial_shapes, level_start_index, mask_flatten
        )

        return encoder_outputs_all, decoder_outputs[:,:bboxes.shape[0]]
