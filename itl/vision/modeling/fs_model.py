import os

import torch
import pytorch_lightning as pl
from transformers import AutoFeatureExtractor, DeformableDetrForObjectDetection


class FewShotSceneGraphGenerator(pl.LightningModule):
    """
    Few-shot visual object detection & class/attribute classification model,
    implemented by attaching lightweight MLP blocks for Deformable DETR model
    outputs. MLP blocks embed DETR output vectors onto a metric space, learned
    by few-shot training with NCA objective.
    """
    def __init__(self, cfg):
        super().__init__()

        detr_model = cfg.vision.model.detr_model
        assets_dir = cfg.paths.assets_dir

        # Load pre-trained deformable DETR to use as basis model
        os.environ["TORCH_HOME"] = os.path.join(assets_dir, "vision_models", "torch")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            detr_model, cache_dir=os.path.join(assets_dir, "vision_models", "detr")
        )
        self.detr = DeformableDetrForObjectDetection.from_pretrained(
            detr_model, cache_dir=os.path.join(assets_dir, "vision_models", "detr")
        )
        if torch.cuda.is_available():
            self.detr.to("cuda")
