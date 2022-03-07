"""
Top vision processing module API that exposes only the high-level functionalities
required by the ITL agent: training (both batch & few-shot mode), inference (full
scene graph generation & classification given bbox), few-shot registration of new
concepts
"""
import os
import pickle
import logging
from itertools import chain

import torch
import detectron2.data.transforms as T
import pytorch_lightning as pl
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from fvcore.common.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message
)

from .data import VGDataModule
from .engine import SceneGraphGenerator
from .utils import has_model, rename_resnet_params
from .utils.path_manager import PathManager

__all__ = ["VisionModule"]


logger = logging.getLogger("vision.module")
logger.setLevel(logging.INFO)

class VisionModule():

    def __init__(self, opts, initial_load=True):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.opts = opts
        self.path_manager = PathManager

        # Fetch detectron config from path, then update corresponding fields with
        # provided command line arguments
        cfg = get_cfg()
        cfg.merge_from_file(opts.config_file_path)

        ### TEMPORARY for search experiments; TODO: erase properly ###
        cfg.MODEL.ROI_HEADS.WEIGHT_EXPONENT = opts.weight_exponent
        cfg.MODEL.ROI_HEADS.WEIGHT_TARGET_MEAN = opts.weight_target_mean

        cfg.MODEL.META = CfgNode()
        cfg.MODEL.META.FEW_SHOT_LOSS_TYPE = opts.few_shot_loss_type
        ### TEMPORARY for search experiments; TODO: erase properly ###

        cfg.SOLVER.MAX_ITER = opts.max_iter
        cfg.OUTPUT_DIR = opts.output_dir_path
        cfg.DATALOADER.NUM_WORKERS = opts.num_dataloader_workers
        cfg = DefaultTrainer.auto_scale_workers(cfg, opts.num_gpus)
        self.cfg = cfg

        # Lists of cls/att/rel category predicates, serving as mapping from category
        # idx to string name
        self.predicates = None

        # pytorch_lightning model; not initialized until self.load_model() or self.train()
        self.model = None

        # Configure pytorch_lightning data module
        augs = [
            T.RandomBrightness(0.9, 1.1),
            T.RandomFlip(prob=0.5),
            T.RandomCrop("absolute", (640, 640))
        ]
        self.dm = VGDataModule(
            cfg, img_count=opts.num_imgs, train_aug_transforms=augs,
            data_dir=opts.data_dir_path
        )
        
        # Configure pytorch_lightning trainer arguments
        self.trainer_args = {
            "strategy": DDPPlugin(find_unused_parameters=True),
            "replace_sampler_ddp": False,
            "num_nodes": 1,
            "gpus": opts.num_gpus,
            "max_epochs": 10 ** 8,
            "max_steps": cfg.SOLVER.MAX_ITER,
            "num_sanity_val_steps": 0,
            "val_check_interval": cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else 10 ** 8,
            "logger": False,
            "default_root_dir": cfg.OUTPUT_DIR,
            "gradient_clip_val": 1.0
        }

        # If initial_load is True, call self.load_model() once with either provided
        # opts.load_checkpoint_path, or checkpoint path provided in default detectron2
        # config (cfg.MODEL.WEIGHTS, as long as it is not empty string)
        if initial_load:
            if opts.load_checkpoint_path is None:
                ckpt_path = cfg.MODEL.WEIGHTS
            else:
                ckpt_path = opts.load_checkpoint_path
            
            if ckpt_path: self.load_model(ckpt_path)

    def train(self, exp_name=None, resume=False, few_shot=None):
        """
        Train the vision model with specified dataset: either in 1) traditional
        batch-mode for training base components, or 2) few-shot mode for training the
        meta-learner component. Not really expected to be called by end-user.

        Args:
            exp_name: str (optional), human-readable display name for this training run; if
                not provided, will be set as random hex string at the beginning of training
            resume: bool (optional), whether to resume training from loaded checkpoint
            few_shot: (int, int, int) tuple (optional), hyperparams defining few-shot learning
                episodes -- (N=# ways, K=# shots, I=# instances per category per episode)
        """
        # Designate datasets to use in config
        self.cfg.DATASETS.TRAIN = ("vg_train",)
        self.cfg.DATASETS.TEST = ("vg_val",)

        # Data preparation step
        self.dm.few_shot = few_shot
        self.dm.prepare_data()
        self.dm.setup("fit")

        # Metadata info loaded
        md = MetadataCatalog.get("vg_train")

        if not self.model:
            # Initialize a model if not exists
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(md.classes)
            self.cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES = len(md.attributes)
            self.cfg.MODEL.ROI_HEADS.NUM_RELATIONS = len(md.relations)

            self.model = SceneGraphGenerator(
                cfg=self.cfg, offline=self.opts.offline
            )
        else:
            # Make sure the model has right number of categories in its RoI head box predictor
            roi_heads = self.model.base_model.roi_heads
            predictor = roi_heads.box_predictor

            reset_predictor = False

            if predictor.num_classes != len(md.classes) or \
                predictor.num_attributes != len(md.attributes) or \
                predictor.num_relations != len(md.relations):

                logger.warning(
                    "Loaded model has different number(s) of categories in its predictor, "
                    "compared to the information in metadata. The box predictor will be replaced "
                    "with a newly initialized classified heads with matching output dims."
                )
                reset_predictor = True
            
            elif self.predicates:
                cat_pairs = chain(
                    zip(self.predicates["cls"], md.classes),
                    zip(self.predicates["att"], md.attributes),
                    zip(self.predicates["rel"], md.relations)
                )
                for c1, c2 in cat_pairs:
                    if c1 != c2:
                        logger.warning(
                            "Loaded model has a different index mapping of categories than that provided "
                            "in metadata. The box predictor will be replaced with a newly initialized "
                            "classified heads with matching output dims."
                        )
                        reset_predictor = True
                        break

            if reset_predictor:
                roi_heads.num_classes = len(md.classes)
                roi_heads.num_attributes = len(md.attributes)
                roi_heads.num_relations = len(md.relations)
                self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(md.classes)
                self.cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES = len(md.attributes)
                self.cfg.MODEL.ROI_HEADS.NUM_RELATIONS = len(md.relations)

                predictor_class = type(predictor)
                roi_heads.box_predictor = predictor_class(
                    self.cfg, roi_heads.box_head.output_shape
                )

        # Check and handle experiment name
        offline_resume_err_msg = \
            "If you want to resume offline training, provide name of the experiment in the " \
            f"output directory ({self.cfg.OUTPUT_DIR}) that contains past TBX logs"

        if self.opts.offline and resume:
            if exp_name is None or \
                not os.path.isdir(os.path.join(self.cfg.OUTPUT_DIR, exp_name, "tbx")):
                raise ValueError(offline_resume_err_msg)

        # Set exp_name as provided if not None, or randomly initialize
        self.model.exp_name = exp_name

        # Add checkpoint callback by appropriately replacing existing ModelCheckpoint
        # callback, or appending to callbacks list if not exists
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc_mean",
            mode="max",
            dirpath=os.path.join(self.cfg.OUTPUT_DIR, self.model.exp_name, "checkpoints"),
            save_last=True
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch}-{step}"
        self.trainer_args["callbacks"] = [checkpoint_callback]

        # Few-shot hparams for setting validation episodes
        self.model.few_shot = few_shot

        # Handle training resumption
        resume_ckpt_path = None
        if resume:
            if self.current_local_ckpt_path:
                resume_ckpt_path = self.current_local_ckpt_path
            else:
                logger.warning(
                    "Tried to resume training but model is not loaded from a pytorch lightning "
                    "checkpoint; training from epoch 0, iteration 0"
                )
            
            if self.current_wandb_id:
                self.model.wandb_id = self.current_wandb_id

        # Start fit
        trainer = pl.Trainer(**self.trainer_args)
        trainer.fit(self.model, self.dm, ckpt_path=resume_ckpt_path)

    @has_model    
    def evaluate(self, few_shot=None):
        """
        Compute evaluation metrics with specified dataset: average precisions for each
        class/attribute category and their means, and image/category-averaged recall@K
        values for each class/attribute/relation category 

        Args:
            few_shot: (int, int, int) tuple (optional), hyperparams defining few-shot learning
                episodes -- (N=# ways, K=# shots, I=# instances per category per episode)
        """
        # Designate datasets to use in config
        self.cfg.DATASETS.TEST = ("vg_test",)

        # Data preparation step
        self.dm.few_shot = few_shot
        self.dm.prepare_data()
        self.dm.setup("test")

        # Few-shot hparams for setting test episodes
        self.model.few_shot = few_shot

        # For logging to W&B
        if self.current_wandb_id:
            self.model.wandb_id = self.current_wandb_id

        # Start test
        trainer = pl.Trainer(**self.trainer_args)
        trainer.test(self.model, self.dm)

    @has_model
    def test(self):
        """
        Interactively test the vision model's behavior on images stored in some directory,
        entering REPL for infinite loop of select-inference-visualize sequence.
        (Mostly for development purpose, maybe removed afterwards...)
        """
        self.model.predicates = self.predicates

        self.trainer_args["enable_progress_bar"] = False

        trainer = pl.Trainer(**self.trainer_args)
        trainer.predict(self.model, self.dm)

    @has_model
    def predict(self):
        """
        Model inference in either one of two modes: 1) full scene graph generation mode,
        where the module is only given an image and needs to return its estimation of
        the full scene graph for the input, or 2) instance classification mode, where
        a number of bboxes are given along with the image and category predictions are
        made for only those instances.

        Args:

        Returns:

        """
        ...

    @has_model
    def add_concept(self):
        """
        Register novel concepts with the few-shot learning capability of the vision module,
        by computing their class codes from given exemplars, while storing feature vectors
        for the exemplars as well, to be ultimately stored in disk storage

        Args:

        """
        ...

    @has_model
    def save_model(self):
        """
        Save current state of the vision model as torch checkpoint, while updating concept
        vocabulary in metadata; this should be called to ensure novel concepts registered
        with add_concept() are permanently learned.

        Args:

        """
        ...
    
    def load_model(self, ckpt_path):
        """
        Load a trained vision model from a torch checkpoint. Should be called before any
        real use of the vision component.

        Args:
            ckpt_path: str, path to checkpoint to load
        """
        assert ckpt_path, "Provided checkpoint path is empty"

        # Clear before update
        self.current_local_ckpt_path = None
        self.current_wandb_id = None

        logger.info("[Checkpointer] Loading from {} ...".format(ckpt_path))
        if not os.path.isfile(ckpt_path):
            local_ckpt_path = self.path_manager.get_local_path(ckpt_path)
            assert os.path.isfile(local_ckpt_path), \
                "Checkpoint {} not found!".format(local_ckpt_path)
        else:
            local_ckpt_path = ckpt_path

        try:
            ckpt = torch.load(local_ckpt_path)
        except RuntimeError:
            with open(local_ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

        if "state_dict" in ckpt:
            # Likely has loaded a checkpoint containing model with this codebase
            params = ckpt["state_dict"]

            # Fetch num categories info from checkpoint and update self.cfg
            num_cls, num_att, num_rel = [
                params[f"base_model.roi_heads.box_predictor.{cat_type}_codes.weight"].shape[0]
                for cat_type in ["cls", "att", "rel"]
            ]
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_cls
            self.cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES = num_att
            self.cfg.MODEL.ROI_HEADS.NUM_RELATIONS = num_rel

            self.model = SceneGraphGenerator.load_from_checkpoint(
                local_ckpt_path, strict=False, cfg=self.cfg, offline=self.opts.offline
            )

            # Store local path of the currently loaded checkpoint, so that it can be
            # fed into self.trainer as ckpt_path arg
            self.current_local_ckpt_path = local_ckpt_path
            if ckpt_path.startswith("wandb://"):
                # Needed if resuming training
                self.current_wandb_id = ckpt_path.strip("wandb://").split("/")[2]
            
            # If checkpoint has "predicates" field, store as property
            if "predicates" in ckpt:
                self.predicates = ckpt["predicates"]

        else:
            # Likely has loaded a pre-trained ResNet from torch hub or detectron2 repo;
            # should map to detectron2 namespace (sort of manually, with regex)
            params = ckpt
            rename_resnet_params(params)

            # Dummy values
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
            self.cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES = 80
            self.cfg.MODEL.ROI_HEADS.NUM_RELATIONS = 80

            # Initialize a new model and load_state_dict from the pretrained params
            self.model = SceneGraphGenerator(
                cfg=self.cfg, offline=self.opts.offline
            )

            keys = self.model.base_model.backbone.load_state_dict(params, strict=False)
            if keys.missing_keys:
                logger.warning(get_missing_parameters_message(keys.missing_keys))
            if keys.unexpected_keys:
                logger.warning(get_unexpected_parameters_message(keys.unexpected_keys))
