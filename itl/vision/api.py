"""
Vision processing module API that exposes only the high-level functionalities
required by the ITL agent: training (both batch & few-shot mode), inference (full
scene graph generation & classification given bbox), few-shot registration of new
concepts
"""
import os
import pickle
import logging
from itertools import chain

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import detectron2.data.transforms as T
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_image_to_rgb
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
from .utils.visualize import visualize_sg_predictions

__all__ = ["VisionModule"]


logger = logging.getLogger("vision.module")
logger.setLevel(logging.INFO)

class VisionModule:

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
        if opts.config_file_path:
            cfg.merge_from_file(opts.config_file_path)

        cfg.MODEL.ROI_HEADS.WEIGHT_EXPONENT = opts.weight_exponent
        cfg.MODEL.ROI_HEADS.WEIGHT_TARGET_MEAN = opts.weight_target_mean

        cfg.SOLVER.MAX_ITER = opts.max_iter
        cfg.OUTPUT_DIR = opts.output_dir_path
        cfg.DATALOADER.NUM_WORKERS = opts.num_dataloader_workers
        cfg = DefaultTrainer.auto_scale_workers(cfg, opts.num_gpus)
        self.cfg = cfg

        # Lists of cls/att/rel category predicates, serving as mapping from category
        # idx to string name
        self.predicates = None
        self.predicates_freq = None

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
        
        # Initialize the category prediction heads (i.e. nuke em) if specified; mainly
        # for experiment purposes, to prepare learners with zero concept knowledge but
        # still with good feature extraction capability
        if opts.initialize_categories:
            roi_heads = self.model.base_model.roi_heads
            predictor_heads = roi_heads.box_predictor

            predictor_heads.num_classes = roi_heads.num_classes = 0
            predictor_heads.num_attributes = roi_heads.num_attributes = 0
            predictor_heads.num_relations = roi_heads.num_relations = 0

            for cat_type in ["cls", "att", "rel"]:
                empty_codes_layer = nn.Linear(
                    predictor_heads.CODE_SIZE, 0, device=self.model.base_model.device
                )
                setattr(predictor_heads, f"{cat_type}_codes", empty_codes_layer)

            self.predicates = {"cls": [], "att": [], "rel": []}
            self.predicates_freq = {"cls": [], "att": [], "rel": []}

        # New visual input buffer
        self.new_input = None

        # Latest raw vision perception, prediction outcomes, feature vectors for RoIs
        self.last_input = None
        self.last_raw = None
        self.last_bboxes = None
        self.scene = None
        self.f_vecs = None

        # Visualized prediction summary (pyplot figure)
        self.summ = None

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
                    "[Vision] Loaded model has different number(s) of categories in its predictor, "
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
                            "[Vision] Loaded model has a different index mapping of categories than "
                            "the mapping provided in metadata. The box predictor will be replaced with "
                            "a newly initialized classified heads with matching output dims."
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
                    "[Vision] Tried to resume training but model is not loaded from a pytorch "
                    "lightning checkpoint; training from epoch 0, iteration 0"
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
    def predict(self, image, exemplars=None, bboxes=None, visualize=False):
        """
        Model inference in either one of two modes: 1) full scene graph generation mode,
        where the module is only given an image and needs to return its estimation of
        the full scene graph for the input, or 2) instance classification mode, where
        a number of bboxes are given along with the image and category predictions are
        made for only those instances.

        Args:
            image: str; input image, passed as path to image file
            bboxes: N*4 array-like (optional); set of bounding boxes provided
            visualize: bool (optional); whether to show visualization of inference result
                on a pop-up window
        Returns:
            raw input image array (C*H*W) processed by the vision module

        """
        self.dm.setup("test")
        self.model.eval()

        # Image data
        if isinstance(image, str):
            inp = { "file_name": image }
            self.last_input = image
        else:
            image = image[:, :, [2,1,0]]
            inp = { "image": image }
            self.last_input = image

        # With/without boxes provided
        if bboxes is None:
            inp = [self.dm.mapper_batch["test"](inp)]
        else:
            inp["annotations"] = bboxes
            inp = [self.dm.mapper_batch["test_props"](inp)]

        with torch.no_grad():
            output, f_vecs = self.model.base_model.inference(inp)
            output = [out["instances"] for out in output]

        f_vecs_org = ({}, {}, {})

        # pred_value_fields = output[0].get_fields()
        pred_value_fields = ["pred_objectness", "pred_boxes"]
        pred_values = zip(*[output[0].get(f) for f in pred_value_fields])

        # Label and reorganize scene graph and feature vectors into a more intelligible
        # format...
        scene = {
            f"o{i}": { f: v.cpu().numpy() for f, v in zip(pred_value_fields, obj) }
            for i, obj in enumerate(pred_values)
        }
        for i, (oi, obj) in enumerate(scene.items()):
            f_vecs_org[0][oi] = f_vecs[0][i]
            f_vecs_org[1][oi] = f_vecs[1][i]
            f_vecs_org[2][oi] = {
                f"o{j}": f_vecs[2][i][j] for j in range(len(scene))
                if oi != f"o{j}"
            }
        #     obj["pred_relations"] = {
        #         f"o{j}": per_obj for j, per_obj in enumerate(obj["pred_relations"])
        #         if oi != f"o{j}"
        #     }
        
        if exemplars is not None:
            # Computing few shot exemplar-based scores
            dev = self.model.base_model.device
            predictor_heads = self.model.base_model.roi_heads.box_predictor
            D = predictor_heads.compress_cls.out_features

            for concept in set(exemplars.pos_exs) | set(exemplars.neg_exs):
                # Prepare values needed to compute distance to pos/neg prototypes (if
                # exemplars are present)
                cat_ind, cat_type = concept

                if concept in exemplars.pos_exs:
                    pos_exs = torch.tensor(exemplars.pos_exs[concept][0], device=dev)
                else:
                    pos_exs = torch.zeros(0, D, device=dev)

                if concept in exemplars.neg_exs:
                    neg_exs = torch.tensor(exemplars.neg_exs[concept][0], device=dev)
                else:
                    neg_exs = torch.zeros(0, D, device=dev)

                if concept in exemplars.pos_exs:
                    pos_proto = pos_exs.mean(dim=0)
                else:
                    pos_proto = None
                
                if concept in exemplars.neg_exs:
                    neg_proto = neg_exs.mean(dim=0)
                else:
                    neg_proto = None

                for oi in scene:
                    if cat_type == "cls":
                        f_vec_cat = f_vecs_org[0][oi]
                    else:
                        pass

                    # Use squared Euclidean distance (L2 norm)
                    if pos_proto is not None:
                        pos_dist = torch.linalg.norm(f_vec_cat-pos_proto).item()
                    else:
                        pos_dist = float("inf")
                    
                    if neg_proto is not None:
                        neg_dist = torch.linalg.norm(f_vec_cat-neg_proto).item()
                    else:
                        neg_dist = float("inf")
                    
                    fs_score = F.softmax(torch.tensor([-pos_dist,-neg_dist]), dim=0)
                    fs_score = fs_score[0].item()

                    if cat_type == "cls":
                        if "pred_classes" in scene[oi]:
                            C = len(scene[oi]["pred_classes"])
                            if cat_ind >= C:
                                scene[oi]["pred_classes"] = np.concatenate((
                                    scene[oi]["pred_classes"], np.zeros(cat_ind+1-C)
                                ))
                        else:
                            scene[oi]["pred_classes"] = np.zeros(cat_ind+1)
                        
                        scene[oi]["pred_classes"][cat_ind] = fs_score
                    else:
                        pass

        # Reformat & resize input image
        img = convert_image_to_rgb(inp[0]["image"].permute(1, 2, 0), "BGR")
        img = cv2.resize(img, dsize=(inp[0]["width"], inp[0]["height"]))

        # Store in case they are needed later
        self.last_raw = img
        self.last_bboxes = [
            {
                "bbox": ent["pred_boxes"],
                "bbox_mode": BoxMode.XYXY_ABS,
                "objectness_scores": ent["pred_objectness"]
            }
            for ent in scene.values()
        ]

        # Store results as state in this vision module
        self.scene = scene
        self.f_vecs = f_vecs_org

        if visualize:
            self.summ = visualize_sg_predictions(img, scene, self.predicates)
    
    def reshow_pred(self):
        assert self.summ is not None, "No predictions have been made yet"
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = self.summ
        self.summ.set_canvas(new_manager.canvas)
        plt.show()

    @has_model
    def add_concept(self, cat_type):
        """
        Register a novel visual concept to the model, expanding the concept inventory of
        corresponding category type (class/attribute/relation). Initialize the new concept's
        category code with a zero vector. Returns the index of the newly added concept.

        Args:
            cat_type: str; either "cls", "att", or "rel", each representing category type
                class, attribute or relation
        Return:
            int index of newly added visual concept
        """
        predictor_heads = self.model.base_model.roi_heads.box_predictor

        cat_predictor = getattr(predictor_heads, f"{cat_type}_codes")

        D = cat_predictor.in_features       # Code dimension
        C = cat_predictor.out_features      # Number of categories (concepts)

        # Expanded category predictor head with novel concept added
        new_cat_predictor = nn.Linear(
            D, C+1, bias=False, device=cat_predictor.weight.device
        )
        new_cat_predictor.weight.data[:C] = cat_predictor.weight.data
        new_cat_predictor.weight.data[C] = 0

        setattr(predictor_heads, f"{cat_type}_codes", new_cat_predictor)

        # Keep category count up to date
        if cat_type == "cls":
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = \
                self.model.base_model.roi_heads.num_classes = \
                predictor_heads.num_classes = C+1
        elif cat_type == "att":
            self.cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES = \
                self.model.base_model.roi_heads.num_attributes = \
                predictor_heads.num_attributes = C+1
        else:
            assert cat_type == "rel"
            self.cfg.MODEL.ROI_HEADS.NUM_RELATIONS = \
                self.model.base_model.roi_heads.num_relations = \
                predictor_heads.num_relations = C+1

        return C
    
    @has_model
    def update_concept(self, concept, ex_vecs, mix_ratio=0.5):
        """
        Update the category code parameter for the designated visual concept in the category
        prediction head, with a set of feature vectors provided as argument.

        Args:
            concept: Concept for which category code vector will be updated
            ex_vecs: numpy.array; set of vector representations of concept exemplars, used
                to update the code vector
            mix_ratio: Mixing ratio between old code vs. new code; 1.0 corresponds to total
                update with new code
        """
        cat_ind, cat_type = concept

        predictor_heads = self.model.base_model.roi_heads.box_predictor
        code_gen = getattr(self.model.meta, f"{cat_type}_code_gen")

        for pol, vecs in ex_vecs.items():
            if pol == "pos":
                target_layer = f"{cat_type}_codes"
            else:
                continue

            if vecs is not None:
                with torch.no_grad():
                    vecs = vecs[0]
                    vecs = torch.tensor(vecs, device=self.model.base_model.device)

                    # Weight average of codes computed from each vector, with heavier
                    # weights placed on more recent records
                    smoothing = len(vecs)             # Additive smoothing parameter
                    weights = torch.arange(
                        1, len(vecs)+1, device=self.model.base_model.device
                    )
                    weights = weights + smoothing
                    new_code = (code_gen(vecs) * weights[:,None]).sum(dim=0)
                    new_code = new_code / weights.sum()
        
                # Fetch category predictor head
                cat_predictor = getattr(predictor_heads, target_layer)

                # Take existing code, to be averaged with new code
                old_code = cat_predictor.weight.data[cat_ind]
                final_code = mix_ratio*new_code + (1-mix_ratio)*old_code

                # Update category code vector
                D = cat_predictor.in_features       # Code dimension
                C = cat_predictor.out_features      # Number of categories (concepts)

                new_cat_predictor = nn.Linear(
                    D, C, bias=False, device=cat_predictor.weight.device
                )
                new_cat_predictor.weight.data = cat_predictor.weight.data
                new_cat_predictor.weight.data[cat_ind] = final_code

                setattr(predictor_heads, target_layer, new_cat_predictor)

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
            ckpt_path: str; path to checkpoint to load
        """
        assert ckpt_path, "Provided checkpoint path is empty"

        # Clear before update
        self.current_local_ckpt_path = None
        self.current_wandb_id = None

        logger.info("[Vision] Loading from {} ...".format(ckpt_path))
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
            
            # If checkpoint has "predicates" & "predicates_freq" field, store as property
            if "predicates" in ckpt:
                self.predicates = ckpt["predicates"]
            if "predicates_freq" in ckpt:
                self.predicates_freq = ckpt["predicates_freq"]

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
