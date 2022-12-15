"""
Vision processing module API that exposes only the high-level functionalities
required by the ITL agent inference (full scene graph generation, classification
given bbox and visual search by concept exemplars). Implemented using publicly
released model of OWL-ViT.
"""
import os
import json
import logging
from PIL import Image
from itertools import product

import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.ops import box_convert, nms, box_area

from .data import FewShotSGGDataModule
from .modeling import FewShotSceneGraphGenerator
from .utils.visualize import visualize_sg_predictions

logger = logging.getLogger(__name__)


WB_PREFIX = "wandb://"
DEF_CON = 0.8       # Default confidence value in absence of binary concept classifier

class VisionModule:

    K = 0               # Top-k detections to leave in ensemble prediction mode
    NMS_THRES = 0.65    # IoU threshold for post-detection NMS

    def __init__(self, cfg):
        self.cfg = cfg

        self.scene = None
        self.f_vecs = None
        self.last_input = None
        self.last_output = None

        self.fs_model_path = None

        # Inventory of distinct visual concepts that the module (and thus the agent
        # equipped with this module) is aware of, one per concept category. Right now
        # I cannot think of any specific type of information that has to be stored in
        # this module (exemplars are stored in long term memory), so let's just keep
        # only integer sizes of inventories for now...
        self.inventories = VisualConceptInventory()

        self.model = FewShotSceneGraphGenerator(self.cfg)

        # Reading W&B config environment variables, if exists
        try:
            load_dotenv(find_dotenv(raise_error_if_not_found=True))
        except OSError as e:
            print(f"While reading dotenv: {e}")

        # If pre-trained vision model is specified, download and load weights
        if "fs_model" in self.cfg.vision.model:
            self.fs_model_path = self.cfg.vision.model.fs_model
            self.load_weights()
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.confusions = set()

    def load_weights(self):
        """ Load model parameter weights from specified source (self.fs_model) """
        # Path to trained weights few-shot prediciton head is provided as either
        # W&B run id or local path to checkpoint file
        assert self.fs_model_path is not None

        if self.fs_model_path.startswith(WB_PREFIX):
            wb_entity = os.environ.get("WANDB_ENTITY")
            wb_project = os.environ.get("WANDB_PROJECT")
            wb_path = self.fs_model_path[len(WB_PREFIX):].split(":")

            if len(wb_path) == 1:
                wb_run_id = wb_path[0]
                wb_alias = "best_k"
            else:
                assert len(wb_path) == 2
                wb_run_id, wb_alias = wb_path

            wb_full_path = f"{wb_entity}/{wb_project}/model-{wb_run_id}:{wb_alias}"
            local_ckpt_path = WandbLogger.download_artifact(
                artifact=wb_full_path,
                save_dir=os.path.join(
                    self.cfg.paths.assets_dir, "vision_models", "wandb", wb_run_id
                )
            )
            os.rename(
                os.path.join(local_ckpt_path, f"model.ckpt"),
                os.path.join(local_ckpt_path, f"model_{wb_alias}.ckpt")
            )
            local_ckpt_path = os.path.join(local_ckpt_path, f"model_{wb_alias}.ckpt")
            logger.info(F"Loading few-shot component weights from {wb_full_path}")
        else:
            local_ckpt_path = self.fs_model_path
            logger.info(F"Loading few-shot component weights from {local_ckpt_path}")

        ckpt = torch.load(local_ckpt_path)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)


    def predict(
        self, image, exemplars, bboxes=None, specs=None, visualize=True, lexicon=None
    ):
        """
        Model inference in either one of three modes:
            1) full scene graph generation mode, where the module is only given
                an image and needs to return its estimation of the full scene
                graph for the input
            2) instance classification mode, where a number of bboxes are given
                along with the image and category predictions are made for only
                those instances
            3) instance search mode, where a specification is provided in the
                form of FOL formula with a variable and best fitting instance(s)
                should be searched

        2) and 3) are 'incremental' in the sense that they should add to an existing
        scene graph which is already generated with some previous execution of this
        method. Provide bboxes arg to run in 2) mode, or spec arg to run in 3) mode.
        """
        if bboxes is None and specs is None:
            assert image is not None    # Image must be provided for ensemble prediction

        if image is not None:
            if isinstance(image, str):
                image = Image.open(image)
            else:
                assert isinstance(image, Image.Image)

        self.model.eval()
        with torch.no_grad():
            # Prediction modes
            if bboxes is None and specs is None:
                # Full (ensemble) prediction
                cls_embeddings, att_embeddings, bboxes_out = self.model(image)
                cls_embeddings = cls_embeddings.cpu().numpy()
                att_embeddings = att_embeddings.cpu().numpy()
                bboxes_out = box_convert(bboxes_out, "xywh", "xyxy")

                self.last_input = image

                # Let's compute 'objectness scores' (with which NMS is run below) by
                # making per-concept prediction on the returned embeddings, and taking
                # max across class concepts. This essentially implements closed-set
                # prediction... Later I could simply add objectness predictor head.
                cls_probs = [None] * self.inventories.cls
                for ci, clf in exemplars.binary_classifiers["cls"].items():
                    if clf is not None:
                        cls_probs[ci] = torch.tensor(
                            clf.predict_proba(cls_embeddings)[:,1]
                        )
                    else:
                        # If binary classifier is None, either positive or negative
                        # exemplar set doesn't exist... Fall back to some default
                        # confidence
                        cls_probs[ci] = torch.full((cls_embeddings.shape[0],), DEF_CON) \
                            if len(exemplars.exemplars_pos["cls"][ci]) > 0 \
                            else torch.full((cls_embeddings.shape[0],), 1-DEF_CON)
                cls_probs = torch.stack(cls_probs, dim=-1)
                objectness_scores = cls_probs.max(dim=-1).values.to(self.model.device)

                # Run NMS and leave top k predictions
                kept_indices = nms(bboxes_out, objectness_scores, self.NMS_THRES)
                topk_inds = [int(i) for i in kept_indices][:self.K]

                # Newly compose a scene graph with the output; filter patches to leave top-k
                # detections
                self.scene = {
                    f"o{i}": {
                        "pred_box": bboxes_out[det_ind].cpu().numpy(),
                        # "pred_objectness": objectness_scores[det_ind].cpu().numpy(),
                        "pred_classes": cls_probs[det_ind].cpu().numpy(),
                        "pred_attributes": np.stack([
                            exemplars.binary_classifiers["att"][ai].predict_proba(
                                att_embeddings[det_ind, None]
                            )[0]
                            if exemplars.binary_classifiers["att"][ai] is not None
                            else (
                                np.array([1.0-DEF_CON, DEF_CON])
                                if len(exemplars.exemplars_pos["att"][ai]) > 0
                                else np.array([DEF_CON, 1.0-DEF_CON])
                            )
                            for ai in range(self.inventories.att)
                        ])[:,1],
                        "pred_relations": {
                            f"o{j}": np.zeros(self.inventories.rel)
                            for j in range(len(topk_inds)) if i != j
                        }
                    }
                    for i, det_ind in enumerate(topk_inds)
                }
                self.f_vecs = {
                    oi: (cls_embeddings[det_ind], att_embeddings[det_ind])
                    for oi, det_ind in zip(self.scene, topk_inds)
                }

                for oi, obj_i in self.scene.items():
                    oi_bb = obj_i["pred_box"]

                    # Relation concepts (Only for "have" concept, manually determined
                    # by the geomtrics of the bounding boxes; note that this is quite
                    # an abstraction. In distant future, relation concepts may also be
                    # open-vocabulary and neurally predicted...)
                    for oj, obj_j in self.scene.items():
                        if oi==oj: continue     # Dismiss self-self object pairs
                        oj_bb = obj_j["pred_box"]

                        x1_int, y1_int, x2_int, y2_int = _box_intersection(oi_bb, oj_bb)

                        bbox_intersection = (x2_int - x1_int) * (y2_int - y1_int) \
                            if x2_int > x1_int and y2_int > y1_int else 0.0
                        bbox2_A = (oj_bb[2] - oj_bb[0]) * (oj_bb[3] - oj_bb[1])

                        obj_i["pred_relations"][oj][0] = bbox_intersection / bbox2_A

            else:
                # Incremental scene graph expansion
                if bboxes is not None:
                    # Instance classification mode
                    bboxes_in = torch.stack([
                        box_convert(torch.tensor(bb["bbox"]), bb["bbox_mode"], "xywh")
                        for bb in bboxes.values()
                    ]).to(self.model.device)
                    cls_embeddings, att_embeddings, bboxes_out = self.model(
                        self.last_input, bboxes_in
                    )
                    incr_cls_embeddings = cls_embeddings[:len(bboxes)].cpu().numpy()
                    incr_att_embeddings = att_embeddings[:len(bboxes)].cpu().numpy()
                    incr_bboxes_out = box_convert(bboxes_out[:len(bboxes)], "xywh", "xyxy")

                else:
                    assert specs is not None
                    # Instance search mode

                    # Mapping between existing entity IDs and their numeric indexing
                    exs_idx_map = { i: ent for i, ent in enumerate(self.scene) }
                    exs_idx_map_inv = { ent: i for i, ent in enumerate(self.scene) }

                    incr_cls_embeddings = []
                    incr_att_embeddings = []
                    incr_bboxes_out = []

                    # Prepare search conditions to feed into model.search() method
                    for s_vars, dscr in specs.values():
                        search_conds = []
                        for d_lit in dscr:
                            conc_type, conc_ind = d_lit.name.split("_")
                            conc_ind = int(conc_ind)

                            if conc_type == "cls" or conc_type == "att":
                                # Fetch set of positive exemplars
                                pos_exs_inds = exemplars.exemplars_pos[conc_type][conc_ind]
                                pos_exs_vecs = exemplars.storage_vec[conc_type][list(pos_exs_inds)]
                                search_conds.append((conc_type, pos_exs_vecs))
                            else:
                                assert conc_type == "rel"

                                # Relations are not neurally predicted, nor used for search
                                # (for now, at least...)
                                continue

                        # Run search method to obtain region proposals
                        proposals, _ = self.model.search(self.last_input, [search_conds], 30)
                        proposals = box_convert(proposals[:,0,:], "xyxy", "xywh")

                        # Predict on the proposals
                        cls_embeddings, att_embeddings, bboxes_out = self.model(
                            self.last_input, proposals, lock_provided_boxes=False
                        )
                        cls_embeddings = cls_embeddings[:len(proposals)].cpu().numpy()
                        att_embeddings = att_embeddings[:len(proposals)].cpu().numpy()
                        bboxes_out = box_convert(bboxes_out[:len(proposals)], "xywh", "xyxy")

                        # Then test each candidate to select the one that's most compatible
                        # to the search spec
                        agg_compatibility_scores = torch.ones(
                            len(proposals), device=self.model.device
                        )
                        for d_lit in dscr:
                            conc_type, conc_ind = d_lit.name.split("_")
                            conc_ind = int(conc_ind)

                            if conc_type == "cls" or conc_type == "att":
                                # Fetch set of positive exemplars
                                clf = exemplars.binary_classifiers[conc_type][conc_ind]
                                if clf is not None:
                                    if conc_type == "cls":
                                        comp_scores = clf.predict_proba(cls_embeddings)[:,1]
                                    else:
                                        comp_scores = clf.predict_proba(att_embeddings)[:,1]

                                    comp_scores = torch.tensor(
                                        comp_scores, device=self.model.device
                                    )
                                else:
                                    comp_scores = torch.full((cls_embeddings.shape[0],), DEF_CON) \
                                        if len(exemplars.exemplars_pos[conc_type][conc_ind]) > 0 \
                                        else torch.full((cls_embeddings.shape[0],), 1-DEF_CON)
                            else:
                                assert conc_type == "rel"

                                # Cannot process relations other than "have" for now...
                                assert conc_ind == 0

                                # Cannot process search specs with more than one variables for
                                # now (not planning to address that for a good while!)
                                assert len(s_vars) == 1

                                # Handles to literal args; either search target variable or
                                # previously identified entity
                                arg_handles = [
                                    ("v", s_vars.index(a[0]))
                                        if a[0] in s_vars
                                        else ("e", exs_idx_map_inv[a[0]])
                                    for a in d_lit.args
                                ]

                                # Bounding boxes for all candidates
                                bboxes_out_A = box_area(bboxes_out)

                                # Fetch bbox of reference entity, against which bbox area
                                # ratios will be calculated among candidates
                                reference_ent = [
                                    arg_ind for arg_type, arg_ind in arg_handles
                                    if arg_type=="e"
                                ][0]
                                reference_ent = exs_idx_map[reference_ent]
                                reference_bbox = self.scene[reference_ent]["pred_box"]
                                reference_bbox = torch.tensor(
                                    reference_bbox, device=self.model.device
                                )

                                # Compute area ratio between the reference box and all patches
                                x1_ints = torch.max(bboxes_out[:,0], reference_bbox[0])
                                y1_ints = torch.max(bboxes_out[:,1], reference_bbox[1])
                                x2_ints = torch.min(bboxes_out[:,2], reference_bbox[2])
                                y2_ints = torch.min(bboxes_out[:,3], reference_bbox[3])
                                intersections = torch.stack([
                                    x1_ints, y1_ints, x2_ints, y2_ints
                                ], dim=-1)
                                ints_invalid = torch.logical_or(
                                    x1_ints > x2_ints, y1_ints > y2_ints
                                )

                                intersections_A = box_area(intersections)
                                intersections_A[ints_invalid] = 0.0

                                comp_scores = intersections_A / bboxes_out_A

                            # Update aggregate compatibility score; using min function
                            # as the t-norm (other options: product, ...)
                            agg_compatibility_scores = torch.minimum(
                                agg_compatibility_scores, comp_scores
                            )

                        # Finally choose and keep the best search output
                        best_match_ind = agg_compatibility_scores.max(dim=0).indices
                        incr_cls_embeddings.append(cls_embeddings[best_match_ind])
                        incr_att_embeddings.append(att_embeddings[best_match_ind])
                        incr_bboxes_out.append(bboxes_out[best_match_ind])

                    incr_cls_embeddings = np.stack(incr_cls_embeddings)
                    incr_att_embeddings = np.stack(incr_att_embeddings)
                    incr_bboxes_out = torch.stack(incr_bboxes_out)

                # Incrementally update the existing scene graph with the output with the
                # detections best complying with the conditions provided
                existing_objs = list(self.scene)
                if bboxes is not None:
                    new_objs = list(bboxes)
                else:
                    new_objs = sum(list(specs), ())

                for oi, oj in product(existing_objs, new_objs):
                    # Add new relation score slots for existing objects
                    self.scene[oi]["pred_relations"][oj] = np.zeros(self.inventories.rel)

                update_data = list(zip(
                    new_objs, incr_cls_embeddings, incr_att_embeddings, incr_bboxes_out
                ))
                for oi, cls_emb, att_emb, bb in update_data:
                    # Register new objects into the existing scene
                    self.scene[oi] = {
                        "pred_box": bb.cpu().numpy(),
                        # "pred_objectness": det_data["score"],
                        "pred_classes": np.stack([
                            exemplars.binary_classifiers["cls"][ci].predict_proba(
                                cls_emb[None]
                            )[0]
                            if exemplars.binary_classifiers["cls"][ci] is not None
                            else (
                                np.array([1.0-DEF_CON, DEF_CON])
                                if len(exemplars.exemplars_pos["cls"][ci]) > 0
                                else np.array([DEF_CON, 1.0-DEF_CON])
                            )
                            for ci in range(self.inventories.cls)
                        ])[:,1],
                        "pred_attributes": np.stack([
                            exemplars.binary_classifiers["att"][ai].predict_proba(
                                att_emb[None]
                            )[0]
                            if exemplars.binary_classifiers["att"][ai] is not None
                            else (
                                np.array([1.0-DEF_CON, DEF_CON])
                                if len(exemplars.exemplars_pos["att"][ai]) > 0
                                else np.array([DEF_CON, 1.0-DEF_CON])
                            )
                            for ai in range(self.inventories.att)
                        ])[:,1],
                        "pred_relations": {
                            **{
                                oj: np.zeros(self.inventories.rel)
                                for oj in existing_objs
                            },
                            **{
                                oj: np.zeros(self.inventories.rel)
                                for oj in new_objs if oi != oj
                            }
                        }
                    }

                for oi in new_objs:
                    oi_bb = self.scene[oi]["pred_box"]

                    # Relation concepts (Within new detections)
                    for oj in new_objs:
                        if oi==oj: continue     # Dismiss self-self object pairs
                        oj_bb = self.scene[oj]["pred_box"]

                        x1_int, y1_int, x2_int, y2_int = _box_intersection(oi_bb, oj_bb)

                        bbox_intersection = (x2_int - x1_int) * (y2_int - y1_int) \
                            if x2_int > x1_int and y2_int > y1_int else 0.0
                        bbox2_A = (oj_bb[2] - oj_bb[0]) * (oj_bb[3] - oj_bb[1])

                        self.scene[oi]["pred_relations"][oj][0] = bbox_intersection / bbox2_A
                    
                    # Relation concepts (Between existing detections)
                    for oj in existing_objs:
                        oj_bb = self.scene[oj]["pred_box"]

                        x1_int, y1_int, x2_int, y2_int = _box_intersection(oi_bb, oj_bb)

                        bbox_intersection = (x2_int - x1_int) * (y2_int - y1_int) \
                            if x2_int > x1_int and y2_int > y1_int else 0.0
                        bbox1_A = (oi_bb[2] - oi_bb[0]) * (oi_bb[3] - oi_bb[1])
                        bbox2_A = (oj_bb[2] - oj_bb[0]) * (oj_bb[3] - oj_bb[1])

                        self.scene[oi]["pred_relations"][oj][0] = bbox_intersection / bbox2_A
                        self.scene[oj]["pred_relations"][oi][0] = bbox_intersection / bbox1_A

                self.f_vecs.update({
                    oi: (cls_emb, att_emb)
                    for oi, cls_emb, att_emb
                    in zip(new_objs, incr_cls_embeddings, incr_att_embeddings)
                })

        if visualize:
            if lexicon is not None:
                lexicon = {
                    conc_type: {
                        ci: lexicon.d2s[(ci, conc_type)][0][0].split("/")[0]
                        for ci in range(getattr(self.inventories, conc_type))
                    }
                    for conc_type in ["cls", "att"]
                }
            self.summ = visualize_sg_predictions(self.last_input, self.scene, lexicon)

    def cache_vectors(self):
        """
        Pre-compute and cache feature vector outputs from the DETR model to speed up
        the training process...
        """
        self.model.eval()

        dataset_path = self.cfg.vision.data.path
        images_path = os.path.join(dataset_path, "images")
        vectors_path = os.path.join(dataset_path, "vectors")
        os.makedirs(vectors_path, exist_ok=True)

        with open(f"{dataset_path}/annotations.json") as ann_f:
            annotations = json.load(ann_f)

        for img in tqdm.tqdm(annotations, total=len(annotations)):
            vec_path = os.path.join(vectors_path, f"{img['file_name']}.vectors")

            if os.path.exists(vec_path): continue
            if len(img["annotations"]) == 0:
                # Still save an empty dict to maintain 1-to-1 file correspondence...
                # for sanity's sake
                torch.save({}, vec_path)
                continue

            image_raw = os.path.join(images_path, img["file_name"])
            image_raw = Image.open(image_raw)
            if image_raw.mode != "RGB":
                # Cast non-RGB images (e.g. grayscale) into RGB format
                old_image_raw = image_raw
                image_raw = Image.new("RGB", old_image_raw.size)
                image_raw.paste(old_image_raw)

            bboxes = [obj["bbox"] for obj in img["annotations"].values()]
            bboxes = torch.tensor(bboxes).to(self.model.device)
            bboxes = box_convert(bboxes, "xywh", "cxcywh")
            bboxes = torch.stack([
                bboxes[:,0] / image_raw.width, bboxes[:,1] / image_raw.height,
                bboxes[:,2] / image_raw.width, bboxes[:,3] / image_raw.height,
            ], dim=-1)

            _, fvecs = self.model.fvecs_from_image_and_bboxes(image_raw, bboxes)
            fvecs = fvecs.cpu()[0]
            fvecs = {oid: fv for oid, fv in zip(img["annotations"], fvecs)}
            torch.save(fvecs, vec_path)

    def train(self):
        """
        Training few-shot visual object detection & class/attribute classification
        model with specified dataset. Uses a pre-trained Deformable DETR as feature
        extraction backbone and learns lightweight MLP blocks (one each for class
        and attribute prediction) for embedding raw feature vectors onto a metric
        space where instances of the same concepts are placed closer. (Mostly likely
        not called by end user.)
        """
        # Prepare DataModule from data config
        dm = FewShotSGGDataModule(self.cfg)

        # Search task -- especially with VG datasets, consisting of bigger images --
        # is quite demanding w.r.t. memory consumption... let's accumulate gradients
        # instead, using the batch size config.
        if self.cfg.vision.task == "fs_search":
            accumulate_grad_batches = self.cfg.vision.data.batch_size
        else:
            accumulate_grad_batches = 1

        # Configure and run trainer
        wb_logger = WandbLogger(
            # offline=True,           # Uncomment for offline run (comment out log_model)
            log_model=True,         # Uncomment for online run (comment out offline)
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            save_dir=self.cfg.paths.outputs_dir
        )
        trainer = pl.Trainer(
            accelerator="auto",
            max_steps=self.cfg.vision.optim.max_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            check_val_every_n_epoch=None,       # Iteration-based val
            log_every_n_steps=self.cfg.vision.optim.log_interval,
            val_check_interval=self.cfg.vision.optim.val_interval,
            num_sanity_val_steps=0,
            logger=wb_logger,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    every_n_train_steps=self.cfg.vision.optim.val_interval,
                    save_last=True
                ),
                LearningRateMonitor(logging_interval='step')
            ]
        )
        trainer.validate(self.model, datamodule=dm)
        trainer.fit(self.model, datamodule=dm)

    def evaluate(self):
        """
        Evaluate best model from a run on test dataset
        """
        # Prepare DataModule from data config
        dm = FewShotSGGDataModule(self.cfg)

        if self.cfg.vision.model.fs_model.startswith(WB_PREFIX):
            wb_logger = WandbLogger(
                # offline=True,           # Uncomment for offline run (comment out log_model)
                project=os.environ.get("WANDB_PROJECT"),
                entity=os.environ.get("WANDB_ENTITY"),
                id=self.cfg.vision.model.fs_model[len(WB_PREFIX):],
                save_dir=self.cfg.paths.outputs_dir,
                resume="must"
            )
            logger = wb_logger
        else:
            logger = False

        trainer = pl.Trainer(accelerator="auto", logger=logger)
        trainer.test(self.model, datamodule=dm)

    def add_concept(self, conc_type):
        """
        Register a novel visual concept to the model, expanding the concept inventory of
        corresponding category type (class/attribute/relation). Note that visual concepts
        are not inseparably attached to some linguistic symbols; such connections are rather
        incidental and should be established independently (consider synonyms, homonyms).
        Plus, this should allow more flexibility for, say, multilingual agents, though there
        is no plan to address that for now...

        Returns the index of the newly added concept.
        """
        C = getattr(self.inventories, conc_type)
        setattr(self.inventories, conc_type, C+1)
        return C


class VisualConceptInventory:
    def __init__(self):
        self.cls = self.att = self.rel = 0


def _box_intersection(box1, box2):
    """ Helper method for obtaining intersection of two boxes (xyxy format) """
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    return x1_int, y1_int, x2_int, y2_int
