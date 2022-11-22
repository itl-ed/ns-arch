import os
import random
from collections import OrderedDict, defaultdict

import torch
import pytorch_lightning as pl
from torchvision.ops import box_convert
from transformers import AutoFeatureExtractor, DeformableDetrForObjectDetection
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrMLPPredictionHead
)

from .detr_abridged import detr_enc_outputs, detr_dec_outputs
from .few_shot import compute_loss_and_metrics


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

        # MLP heads to attach on top of encoder & embedder outputs for metric-based
        # few-shot search (conditioned detection)
        self.fs_spec_fuse = DeformableDetrMLPPredictionHead(
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
        if "task" in self.cfg.vision:
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
                for prm in self.fs_spec_fuse.parameters():
                    prm.requires_grad = True
                for prm in self.fs_search_bbox.parameters():
                    prm.requires_grad = True
            else:
                raise ValueError("Invalid task type for training")
        
        self.save_hyperparameters()

    def training_step(self, batch, *_):
        if isinstance(batch, tuple):
            # Simpler case of single batch from single dataloader
            loss, metrics, _, _ = self._process_batch(batch)

            self.log("train_loss", loss.item())
            for metric, val in metrics.items():
                self.log(f"train_{metric}", val)

            return loss
        else:
            # 'Batch' consists of batches from multiple dataloaders (most likely
            # in fs_search task mode); process and log each
            loss_total = 0
            for b in batch:
                loss, metrics, _, _ = self._process_batch(b)
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
        loss, metrics, _, _ = self._process_batch(batch)
        return loss, metrics, batch[1]

    def validation_epoch_end(self, outputs):
        if len(self.trainer.val_dataloaders) == 1:
            outputs = [outputs]

        avg_losses = []
        for outputs_per_dataloader in outputs:
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
        _, metrics, embeddings, labels = self._process_batch(batch)

        for metric, val in metrics.items():
            self.log(f"test_{metric}_{batch[1]}", val, batch_size=len(batch[0][0]))

        return embeddings, labels
    
    def test_epoch_end(self, outputs):
        if self.cfg.vision.task.pred_type == "fs_classify" and \
            self.logger is not None and hasattr(self.logger, "log_table"):

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

    def _process_batch(self, batch):
        """
        Shared subroutine for processing batch to obtain loss & performance metric
        """
        self.detr.eval()        # DETR always eval mode

        batch_data, conc_type = batch

        images = batch_data["image"]
        bboxes = batch_data["bboxes"]
        bb_inds = batch_data["bb_inds"]
        conc_labels = batch_data["concept_label"]

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
        N = len(set(conc_labels))
        K = B // N

        # DETR Decoder output vectors, either computed from scratch or cached values
        # retrieved from disk
        detr_enc_outs = []; detr_dec_outs = []
        if "dec_out_cached" in batch_data:
            # Decoder outputs are pre-computed and retrieved
            detr_dec_outs = batch_data["dec_out_cached"]

            if self.cfg.vision.task.pred_type == "fs_search":
                # Few-shot search task needs encoder outputs (per-pixel vectors)
                assert images is not None
                for img in images:
                    enc_out = detr_enc_outputs(
                        self.detr, img, self.feature_extractor
                    )
                    detr_enc_outs.append(enc_out)
        else:
            # Need to compute from images in batch

            # Process each image with DETR - one-by-one
            for img, bbs, bbis in zip(images, bboxes, bb_inds):
                enc_out, dec_out = self.fvecs_from_image_and_bboxes(img, bbs)
                detr_dec_outs.append(dec_out[:,bbis])

                if self.cfg.vision.task.pred_type == "fs_search":
                    # Few-shot search task needs encoder outputs (per-pixel vectors)
                    detr_enc_outs.append(enc_out)

            detr_dec_outs = torch.cat(detr_dec_outs)

        # Search targets in few-shot search mode
        if self.cfg.vision.task.pred_type == "fs_search":
            bboxes_search_targets = [
                bbs[bbis] for bbs, bbis in zip(bboxes, batch_data["bb_all"])
            ]
        else:
            bboxes_search_targets = None

        # Computing appropriate loss & metric values according to task specification
        loss, metrics, embeddings = compute_loss_and_metrics(
            self, self.cfg.vision.task.pred_type, conc_type,
            detr_enc_outs, detr_dec_outs, B, N, K, bboxes_search_targets
        )

        return loss, metrics, embeddings, conc_labels

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

        decoder_outputs = detr_dec_outputs(
            self.detr, encoder_outputs, bboxes,
            valid_ratios, spatial_shapes, level_start_index, mask_flatten
        )

        return encoder_outputs_all, decoder_outputs
