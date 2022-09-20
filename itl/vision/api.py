"""
Vision processing module API that exposes only the high-level functionalities
required by the ITL agent: training (both batch & few-shot mode), inference (full
scene graph generation & classification given bbox), few-shot registration of new
concepts
"""
import os
import copy
import logging
from itertools import chain, permutations, product

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
from .utils.visualize import visualize_sg_predictions

__all__ = ["VisionModule"]


logger = logging.getLogger("vision.module")
logger.setLevel(logging.INFO)

class VisionModule:

    def __init__(self, opts):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.opts = opts

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

        # New visual input buffer
        self.new_input = None

        # Latest raw vision perception, prediction outcomes, cached feature vectors
        self.last_input = None
        self.last_raw = None
        self.scene = None
        self.f_vecs = None

        # Visualized prediction summary (pyplot figure)
        self.summ = None

    def train(self, exp_name=None, resume=False, few_shot=None):
        """
        Train the vision model with specified dataset: either in 1) traditional
        batch-mode for training base components, or 2) few-shot mode for training the
        meta-learner component. Not really expected to be called by lay end-user.

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
    def predict(self, image, exemplars=None, bboxes=None, specs=None, visualize=False):
        """
        Model inference in either one of three modes:
            1) full scene graph generation mode, where the module is only given an image
                and needs to return its estimation of the full scene graph for the input
            2) instance classification mode, where a number of bboxes are given along
                with the image and category predictions are made for only those instances
            3) instance search mode, where a specification is provided in the form of FOL
                formula with a variable and best fitting instance(s) should be searched

        2) and 3) are 'incremental' in the sense that they should add to an existing scene
        graph which is already generated with some previous execution of this method. Provide
        bboxes arg to run in 2) mode, or spec arg to run in 3) mode.

        Args:
            image: str; input image, passed as path to image file
            exemplars: Exemplars (optional); set of positive & negative concept exemplars
            bboxes: dict[str, dict] (optional); set of entities with bbox info
            specs: dict[tuple[str], (list[str], frozenset[Literal])] (optional); set of FOL
                search specifications
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

        # Prediction modes
        if bboxes is None and specs is None:
            # Full (ensemble) prediction
            inp = [self.dm.mapper_batch["test"](inp)]
            exs_cached = exs_idx_map = inc_idx_map = search_specs = None
        else:
            # Incremental prediction; fetch bboxes and box f_vecs for existing detections
            exs_cached = {
                "backbone_output": self.f_vecs[5],
                "detections": [
                    {
                        "bbox": self.scene[ent]["pred_boxes"],
                        "box_f_vec": self.f_vecs[3][ent],
                        "sem_f_vec": self.f_vecs[4][ent],
                    }
                    for ent in self.scene
                ]
            }
            exs_idx_map = { i: ent for i, ent in enumerate(self.scene) }
            exs_idx_map_inv = { ent: i for i, ent in enumerate(self.scene) }

            if bboxes is not None:
                # Instance classification mode
                inp["annotations"] = list(bboxes.values())
                inp = [self.dm.mapper_batch["test_props"](inp)]

                inc_idx_map = { i: ent for i, ent in enumerate(bboxes) }
                search_specs = None
            else:
                assert specs is not None
                # Instance search mode
                inp = [self.dm.mapper_batch["test"](inp)]

                oi_offsets = np.cumsum([0]+[len(ents) for ents in specs][:-1])
                inc_idx_map = {
                    offset+i: e
                    for ents, offset in zip(specs, oi_offsets)
                    for i, e in enumerate(ents)
                }

                # Provide search specs as appropriate vectors; required info provided
                # differently, depending on whether prediction is exemplar-based or
                # model-based
                if exemplars is None:
                    # Search spec info for model-based prediction
                    search_by = "model"
                    raise NotImplementedError
                else:
                    # Search spec info for exemplar-based prediction
                    search_by = "exemplar"
                    search_conds = []
                    for s_vars, dscr in specs.values():
                        dscr_translated = []
                        for d_lit in dscr:
                            cat_type, conc_ind = d_lit.name.split("_")
                            conc_ind = int(conc_ind)

                            # Handles to literal args; either search target variable
                            # or previously identified entity
                            arg_handles = [
                                ("v", s_vars.index(a[0]))
                                    if a[0] in s_vars
                                    else ("e", exs_idx_map_inv[a[0]])
                                for a in d_lit.args
                            ]
                            # Reference vector, as mean of positive exemplar feature vectors
                            # fetched from exemplar base
                            ex_vecs = exemplars[(conc_ind, cat_type)]
                            ref_vec_pos = torch.tensor(
                                ex_vecs["pos"], device=self.model.base_model.device
                            ).mean(dim=0) if len(ex_vecs["pos"]) > 0 else None
                            ref_vec_neg = torch.tensor(
                                ex_vecs["neg"], device=self.model.base_model.device
                            ).mean(dim=0) if len(ex_vecs["neg"]) > 0 else None
                            ref_vecs = (ref_vec_pos, ref_vec_neg)

                            dscr_translated.append((cat_type, arg_handles, ref_vecs))

                        search_conds.append((len(s_vars), dscr_translated))
                    
                    search_specs = (search_by, search_conds)

        with torch.no_grad():
            output, f_vecs, inc_out = self.model.base_model.inference(
                inp, exs_cached=exs_cached, search_specs=search_specs
            )
            output = [out["instances"] for out in output]

        # pred_value_fields = output[0].get_fields()
        pred_value_fields = ["pred_objectness", "pred_boxes"]
        pred_values = zip(*[output[0].get(f) for f in pred_value_fields])

        if inc_out is None:
            ## Newly generate a scene graph with the output

            self.f_vecs = ({}, {}, {}, {}, {}, f_vecs[5])    # f_vecs[5]: backbone output

            # Label & reorganize scene graph, and feature vectors into a more intelligible
            # format...
            self.scene = {
                f"o{i}": { f: v.cpu().numpy() for f, v in zip(pred_value_fields, obj) }
                for i, obj in enumerate(pred_values)
            }
            # Filter by objectness threshold
            self.scene = {
                oi: obj for oi, obj in self.scene.items() if obj["pred_objectness"] > 0.5
            }
            for i, (oi, obj) in enumerate(self.scene.items()):
                self.f_vecs[0][oi] = f_vecs[0][i]       # cls_f_vecs
                self.f_vecs[1][oi] = f_vecs[1][i]       # att_f_vecs

                self.f_vecs[2][oi] = {                  # rel_f_vecs
                    f"o{j}": f_vecs[2][i][j] for j in range(len(self.scene))
                    if oi != f"o{j}"
                }

                self.f_vecs[3][oi] = f_vecs[3][i]       # box_f_vecs
                self.f_vecs[4][oi] = f_vecs[4][i]       # sem_f_vecs
            
            #     obj["pred_relations"] = {
            #         f"o{j}": per_obj for j, per_obj in enumerate(obj["pred_relations"])
            #         if oi != f"o{j}"
            #     }

            # Nodes and edges in scene graphs for which few-shot predictions should be made
            fs_pred_nodes = list(self.scene.keys())
            fs_pred_edges = list(permutations(self.scene, 2))

            # Reformat & resize input image
            img = convert_image_to_rgb(inp[0]["image"].permute(1, 2, 0), "BGR")
            img = cv2.resize(img, dsize=(inp[0]["width"], inp[0]["height"]))

            # Store in case they are needed later
            self.last_raw = img

        else:
            ## Incrementally update the existing scene graph with the output

            # Recover incrementally obtained relation feature vectors, i.e. top right and
            # bottom left parts of 2-by-2 partitioning
            N_E = len(exs_idx_map); N_N = len(inc_idx_map)
            D = inc_out[1].shape[-1]
            inc_rel_f_vecs = inc_out[1].view(-1, 2, D)
            rel_f_vecs_top_right = inc_rel_f_vecs[:,0,:].view(N_E, N_N, -1)
            rel_f_vecs_bottom_left = inc_rel_f_vecs[:,1,:].view(N_N, N_E, -1)

            # Update cached scene graph and feature vectors
            for i, obj in enumerate(pred_values):
                oi = inc_idx_map[i]
                self.scene[oi] = { f: v.cpu().numpy() for f, v in zip(pred_value_fields, obj) }

                self.f_vecs[0][oi] = f_vecs[0][i]       # cls_f_vecs
                self.f_vecs[1][oi] = f_vecs[1][i]       # att_f_vecs

                self.f_vecs[2][oi] = {                  # rel_f_vecs
                    exs_idx_map[j]: rfv
                    for j, rfv in enumerate(rel_f_vecs_bottom_left[i])
                }
                for j, rfv in enumerate(rel_f_vecs_top_right[:,i,:]):
                    oj = exs_idx_map[j]
                    self.f_vecs[2][oj][oi] = rfv
                for j in range(N_N):
                    if i != j:
                        oj = inc_idx_map[j]
                        self.f_vecs[2][oi][oj] = f_vecs[2][i][j]

                self.f_vecs[3][oi] = f_vecs[3][i]       # box_f_vecs
                self.f_vecs[4][oi] = f_vecs[4][i]       # sem_f_vecs

            # Nodes and edges in scene graphs for which few-shot predictions should be made
            fs_pred_nodes = list(inc_idx_map.values())
            fs_pred_edges = list(chain(
                permutations(fs_pred_nodes, 2),
                product(exs_idx_map.values(), inc_idx_map.values()),
                product(inc_idx_map.values(), exs_idx_map.values()),
            ))

        if exemplars is not None:
            # Computing few shot exemplar-based scores
            dev = self.model.base_model.device
            predictor_heads = self.model.base_model.roi_heads.box_predictor
            D = predictor_heads.compress_cls.out_features

            for cat_type in ["cls", "att", "rel"]:
                pos_exs = exemplars.exemplars_pos[cat_type]
                neg_exs = exemplars.exemplars_neg[cat_type]

                for conc_ind in (set(pos_exs) | set(neg_exs)):
                    # Prepare values needed to compute distance to pos/neg prototypes
                    # (if exemplars are present; some day we could maybe try zero-shot
                    # prototype estimation by leveraging other resources like pre-trained
                    # word embeddings?)
                    if conc_ind in pos_exs and len(pos_exs[conc_ind])>0:
                        proto_pos = exemplars.storage_vec[cat_type][list(pos_exs[conc_ind])]
                        proto_pos = torch.tensor(proto_pos, device=dev).mean(dim=0)
                    else:
                        proto_pos = None
                    
                    if conc_ind in neg_exs and len(neg_exs[conc_ind])>0:
                        proto_neg = exemplars.storage_vec[cat_type][list(neg_exs[conc_ind])]
                        proto_neg = torch.tensor(proto_neg, device=dev).mean(dim=0)
                    else:
                        proto_neg = None

                    # Fetch appropriate feature vector and name of prediction field to fill
                    if cat_type == "cls" or cat_type == "att":
                        # Class and attribute predictions for scene graph nodes
                        for oi in fs_pred_nodes:
                            if cat_type == "cls":
                                f_vec_cat = self.f_vecs[0][oi]
                                field_name = "pred_classes"
                            else:
                                f_vec_cat = self.f_vecs[1][oi]
                                field_name = "pred_attributes"

                            # Use squared Euclidean distance (L2 norm)
                            if proto_pos is not None:
                                pos_dist = torch.linalg.norm(f_vec_cat-proto_pos).item()
                            else:
                                pos_dist = float("inf")
                            
                            if proto_neg is not None:
                                neg_dist = torch.linalg.norm(f_vec_cat-proto_neg).item()
                            else:
                                neg_dist = float("inf")

                            fs_score = F.softmax(torch.tensor([-pos_dist,-neg_dist]), dim=0)
                            fs_score = fs_score[0].item()

                            # Fill in the scene graph with the score
                            if field_name in self.scene[oi]:
                                C = len(self.scene[oi][field_name])
                                if conc_ind >= C:
                                    self.scene[oi][field_name] = np.concatenate((
                                        self.scene[oi][field_name], np.zeros(conc_ind+1-C)
                                    ))
                            else:
                                self.scene[oi][field_name] = np.zeros(conc_ind+1)
                            
                            self.scene[oi][field_name][conc_ind] = fs_score

                    else:
                        # Relation predictions for scene graph edges
                        assert cat_type == "rel"
                        for oi, oj in fs_pred_edges:
                            f_vec_cat = self.f_vecs[2][oi][oj]
                            field_name = "pred_relations"

                            # Use squared Euclidean distance (L2 norm)
                            if proto_pos is not None:
                                pos_dist = torch.linalg.norm(f_vec_cat-proto_pos).item()
                            else:
                                pos_dist = float("inf")
                            
                            if proto_neg is not None:
                                neg_dist = torch.linalg.norm(f_vec_cat-proto_neg).item()
                            else:
                                neg_dist = float("inf")

                            fs_score = F.softmax(torch.tensor([-pos_dist,-neg_dist]), dim=0)
                            fs_score = fs_score[0].item()

                            # Fill in the scene graph with the score
                            if field_name in self.scene[oi]:
                                if oj in self.scene[oi][field_name]:
                                    C = len(self.scene[oi][field_name][oj])
                                    if conc_ind >= C:
                                        self.scene[oi][field_name][oj] = np.concatenate((
                                            self.scene[oi][field_name][oj], np.zeros(conc_ind+1-C)
                                        ))
                                else:
                                    self.scene[oi][field_name][oj] = np.zeros(conc_ind+1)
                            else:
                                self.scene[oi][field_name] = { oj: np.zeros(conc_ind+1) }
                            
                            self.scene[oi][field_name][oj][conc_ind] = fs_score

        if visualize:
            self.summ = visualize_sg_predictions(
                self.last_raw, self.scene, self.predicates
            )

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
            mix_ratio: float; Mixing ratio between old code vs. new code - 1.0 corresponds to
                total update with new code
        """
        conc_ind, cat_type = concept

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
                old_code = cat_predictor.weight.data[conc_ind]
                final_code = mix_ratio*new_code + (1-mix_ratio)*old_code

                # Update category code vector
                D = cat_predictor.in_features       # Code dimension
                C = cat_predictor.out_features      # Number of categories (concepts)

                new_cat_predictor = nn.Linear(
                    D, C, bias=False, device=cat_predictor.weight.device
                )
                new_cat_predictor.weight.data = cat_predictor.weight.data
                new_cat_predictor.weight.data[conc_ind] = final_code

                setattr(predictor_heads, target_layer, new_cat_predictor)
    
    def load_model(self, ckpt, ckpt_path=None, local_ckpt_path=None):
        """
        Load a trained vision model from a torch checkpoint. Should be called before
        any real use of the vision component. The optional path arguments are needed
        when resuming batch training the model.

        Args:
            ckpt: dict; loaded checkpoint object
            ckpt_path: (Optional) str; original checkpoint path str
            local_ckpt_path: (Optional) str; resolved local checkpoint path str
        """
        assert ckpt and len(ckpt)>0, "Provided checkpoint is empty"

        # Clear before update
        self.current_local_ckpt_path = None
        self.current_wandb_id = None

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

            self.model = SceneGraphGenerator.load_from_dict(
                ckpt, strict=False, cfg=self.cfg, offline=self.opts.offline
            )

            # Store local path of the currently loaded checkpoint, so that it can be
            # fed into self.trainer as ckpt_path arg
            self.current_local_ckpt_path = local_ckpt_path
            if ckpt_path is not None and ckpt_path.startswith("wandb://"):
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

        # Initialize the category prediction heads (i.e. nuke'em) if specified; mainly
        # for experiment purposes, to prepare learners with zero concept knowledge but
        # still with good feature extraction capability
        if self.opts.initialize_categories:
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
