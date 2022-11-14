"""
Vision processing module API that exposes only the high-level functionalities
required by the ITL agent inference (full scene graph generation, classification
given bbox and visual search by concept exemplars). Implemented using publicly
released model of OWL-ViT.
"""
import os
import json
from PIL import Image
from itertools import product

import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.ops import box_convert, nms, box_iou, box_area

from .data import FewShotSGGDataModule
from .modeling import FewShotSceneGraphGenerator
from .utils.visualize import visualize_sg_predictions


WB_PREFIX = "wandb://"

class VisionModule:

    K = 0               # Top-k detections to leave in ensemble prediction mode
    NMS_THRES = 0.5     # IoU threshold for post-detection NMS

    def __init__(self, cfg):
        self.cfg = cfg

        self.scene = None
        self.f_vecs = None
        self.last_input = None
        self.last_output = None

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
            # Path to trained weights few-shot prediciton head, provided as either
            # W&B run id or local path to checkpoint file
            if self.cfg.vision.model.fs_model.startswith(WB_PREFIX):
                wb_entity = os.environ.get("WANDB_ENTITY")
                wb_project = os.environ.get("WANDB_PROJECT")
                wb_run_id = self.cfg.vision.model.fs_model[len(WB_PREFIX):]

                local_ckpt_path = WandbLogger.download_artifact(
                    artifact=f"{wb_entity}/{wb_project}/model-{wb_run_id}:best_k",
                    save_dir=os.path.join(
                        self.cfg.paths.assets_dir, "vision_models", "wandb", wb_run_id
                    )
                )
                local_ckpt_path = os.path.join(local_ckpt_path, "model.ckpt")
            else:
                local_ckpt_path = self.cfg.vision.model.fs_model

            ckpt = torch.load(local_ckpt_path)
            self.model.load_state_dict(ckpt["state_dict"], strict=False)

    def predict(
        self, image, label_texts=None, label_exemplars=None,
        bboxes=None, specs=None, visualize=True, lexicon=None
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
        # Must provide either set of label texts, or set of exemplars of concepts
        assert label_texts is not None or label_exemplars is not None
        if bboxes is None and specs is None:
            assert image is not None    # Image must be provided for ensemble prediction

        with torch.no_grad():
            # Prediction modes
            if bboxes is None and specs is None:
                # Full (ensemble) prediction
                results, output, input_img, label_concepts = self.owlvit_process(
                    image, label_texts=label_texts, label_exemplars=label_exemplars
                )
                self.last_input = input_img
                self.last_output = (results, output, label_concepts)

                label_concepts_per_catType = {
                    cat_type: {
                        conc_ind: i
                        for i, (conc_ind, this_cat_type) in enumerate(label_concepts)
                        if this_cat_type==cat_type
                    }
                    for cat_type in ["cls", "att"]
                }

                patch_boxes = results[0]["boxes"]
                patch_embs = output.class_embeds[0]

                # They say we don't need NMS with OwL-ViT, but it turns out there are many
                # duplicate detection boxes when ranked by objectness score. Maybe they meant
                # NMS is not needed for training... Anyhow, here goes NMS
                kept_indices = nms(patch_boxes, results[0]["scores"], self.NMS_THRES)

                # Newly compose a scene graph with the output; filter patches to leave top-k
                # detections
                topk_inds = [int(i) for i in kept_indices][:self.K]
                topk_detections = [
                    {
                        "bbox": patch_boxes[i].cpu().numpy(),
                        "score": float(results[0]["scores"][i]),
                        "conc_pos_logits": output.logits[0,i,::2].cpu().numpy(),
                        "conc_neg_logits": output.logits[0,i,1::2].cpu().numpy()
                    }
                    for i in topk_inds
                ]
                self.scene = {
                    f"o{i}": {
                        "pred_boxes": det_data["bbox"],
                        "pred_objectness": det_data["score"],
                        "pred_classes": np.zeros(self.inventories.cls),
                        "pred_attributes": np.zeros(self.inventories.att),
                        "pred_relations": {
                            f"o{j}": np.zeros(self.inventories.rel)
                            for j in range(len(topk_detections)) if i != j
                        }
                    }
                    for i, det_data in enumerate(topk_detections)
                }
                self.f_vecs = {
                    oi: patch_embs[i].cpu().numpy()
                    for oi, i in zip(self.scene, topk_inds)
                }

                # Fill in per-concept scores from logits
                for (oi, obj), det_data in zip(self.scene.items(), topk_detections):
                    # Class concepts
                    cls_logits = {
                        conc_ind: (
                            det_data["conc_pos_logits"][i],
                            det_data["conc_neg_logits"][i]
                        )
                        for conc_ind, i in label_concepts_per_catType["cls"].items()
                    }
                    for conc_ind, (p_logit, n_logit) in cls_logits.items():
                        score = F.softmax(torch.tensor([p_logit, n_logit]))[0].item()
                        obj["pred_classes"][conc_ind] = score

                    # Attribute concepts
                    att_logits = {
                        conc_ind: (
                            det_data["conc_pos_logits"][i],
                            det_data["conc_neg_logits"][i]
                        )
                        for conc_ind, i in label_concepts_per_catType["att"].items()
                    }
                    for conc_ind, (p_logit, n_logit) in att_logits.items():
                        score = F.softmax(torch.tensor([p_logit, n_logit]))[0].item()
                        obj["pred_attributes"][conc_ind] = score

                    # Relation concepts (Only for "have" concept, manually determined
                    # by the geomtrics of the bounding boxes; note that this is quite
                    # an abstraction. In distant future, relation concepts may also be
                    # open-vocabulary and neurally predicted...)
                    for oj, det_data2 in zip(self.scene, topk_detections):
                        if oi==oj: continue     # Dismiss self-self object pairs
                        x1_int, y1_int, x2_int, y2_int = _box_intersection(
                            det_data["bbox"], det_data2["bbox"]
                        )

                        bbox_intersection = (x2_int - x1_int) * (y2_int - y1_int) \
                            if x2_int > x1_int and y2_int > y1_int else 0.0
                        bbox2_A = (det_data2["bbox"][2] - det_data2["bbox"][0]) * \
                            (det_data2["bbox"][3] - det_data2["bbox"][1])

                        obj["pred_relations"][oj][0] = bbox_intersection / bbox2_A

            else:
                # Incremental scene graph expansion
                results, output, label_concepts = self.last_output

                label_concepts_per_catType = {
                    cat_type: {
                        conc_ind: i
                        for i, (conc_ind, this_cat_type) in enumerate(label_concepts)
                        if this_cat_type==cat_type
                    }
                    for cat_type in ["cls", "att"]
                }

                if bboxes is not None:
                    # Instance classification mode

                    # Find existing detections with highest overlap (IoU) w.r.t. the
                    # provided bounding boxes, disregarding their objectness scores
                    all_bboxes = torch.stack([
                        torch.tensor(obj["bbox"]) for obj in bboxes.values()
                    ]).to(device=results[0]["boxes"].device)
                    ious = box_iou(all_bboxes, results[0]["boxes"])

                    best_indices = ious.max(dim=1).indices

                else:
                    assert specs is not None
                    # Instance search mode

                    # Mapping between existing entity IDs and their numeric indexing
                    exs_idx_map = { i: ent for i, ent in enumerate(self.scene) }
                    exs_idx_map_inv = { ent: i for i, ent in enumerate(self.scene) }

                    # Cast search specs into appropriate testing criteria and find the
                    # best matching image patches
                    best_indices = []
                    for s_vars, dscr in specs.values():
                        # Per-candidate "compatibility scores" for each search spec, to be
                        # aggregated at the end
                        patch_compatibilities = []

                        for d_lit in dscr:
                            cat_type, conc_ind = d_lit.name.split("_")
                            conc_ind = int(conc_ind)

                            if cat_type == "cls" or cat_type == "att":
                                # For now we will only consider cases where ensemble prediction
                                # has been made with all known concepts as query labels already,
                                # and additional concept tests are not needed. This may have to
                                # be relaxed later for cases when inventories of visual concepts
                                # grow too large and some sort of restriction of query label space
                                # has to happen (i.e. don't consider parts, rare concepts, etc.).
                                assert conc_ind in label_concepts_per_catType[cat_type]

                                query_ind = label_concepts_per_catType[cat_type][conc_ind]

                                conc_pos_logits = output.logits[0,:,::2][:,query_ind]
                                conc_neg_logits = output.logits[0,:,1::2][:,query_ind]

                                comp_scores = torch.stack([conc_pos_logits, conc_neg_logits])
                                comp_scores = F.softmax(comp_scores, dim=0)[0]
                            else:
                                # Cannot process relations other than "have" for now...
                                assert cat_type == "rel" and conc_ind == 0

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
                                cand_bboxes = results[0]["boxes"]
                                cand_bboxes_A = box_area(cand_bboxes)

                                # Fetch bbox of reference entity, against which bbox area
                                # ratios will be calculated among candidates
                                reference_ent = [
                                    arg_ind for arg_type, arg_ind in arg_handles
                                    if arg_type=="e"
                                ][0]
                                reference_ent = exs_idx_map[reference_ent]
                                reference_bbox = self.scene[reference_ent]["pred_boxes"]
                                reference_bbox = torch.tensor(
                                    reference_bbox, device=cand_bboxes.device
                                )

                                # Compute IoUs between the reference box and all patches
                                x1_ints = torch.max(cand_bboxes[:,0], reference_bbox[0])
                                y1_ints = torch.max(cand_bboxes[:,1], reference_bbox[1])
                                x2_ints = torch.min(cand_bboxes[:,2], reference_bbox[2])
                                y2_ints = torch.min(cand_bboxes[:,3], reference_bbox[3])
                                cand_intersections = torch.stack([
                                    x1_ints, y1_ints, x2_ints, y2_ints
                                ], dim=-1)
                                ints_invalid = torch.logical_or(
                                    x1_ints > x2_ints, y1_ints > y2_ints
                                )

                                cand_intersections_A = box_area(cand_intersections)
                                cand_intersections_A[ints_invalid] = 0.0

                                comp_scores = cand_intersections_A / cand_bboxes_A

                            # Collect compatibility scores obtained
                            patch_compatibilities.append(comp_scores)

                        # Aggregate compatibility scores as product across search specs
                        patch_compatibilities = torch.stack(patch_compatibilities).prod(dim=0)

                        # Append best candidate for this search spec
                        best_indices.append(patch_compatibilities.max(dim=0).indices.item())

                # Incrementally update the existing scene graph with the output with the
                # detections best complying with the conditions provided
                best_detections = [
                    {
                        "bbox": results[0]["boxes"][i].cpu().numpy(),
                        "score": float(results[0]["scores"][i]),
                        "conc_pos_logits": output.logits[0,i,::2].cpu().numpy(),
                        "conc_neg_logits": output.logits[0,i,1::2].cpu().numpy()
                    }
                    for i in best_indices
                ]

                existing_objs = list(self.scene)
                if bboxes is not None:
                    new_objs = list(bboxes)
                else:
                    new_objs = sum(list(specs), ())

                for oi, oj in product(existing_objs, new_objs):
                    # Add new relation score slots for existing objects
                    self.scene[oi]["pred_relations"][oj] = np.zeros(self.inventories.rel)

                for oi, det_data in zip(new_objs, best_detections):
                    # Register new objects into the existing scene
                    self.scene[oi] = {
                        "pred_boxes": det_data["bbox"],
                        "pred_objectness": det_data["score"],
                        "pred_classes": np.zeros(self.inventories.cls),
                        "pred_attributes": np.zeros(self.inventories.att),
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
                    obj = self.scene[oi]

                    # Class concepts
                    cls_logits = {
                        conc_ind: (
                            det_data["conc_pos_logits"][i],
                            det_data["conc_neg_logits"][i]
                        )
                        for conc_ind, i in label_concepts_per_catType["cls"].items()
                    }
                    for conc_ind, (p_logit, n_logit) in cls_logits.items():
                        score = F.softmax(torch.tensor([p_logit, n_logit]))[0].item()
                        obj["pred_classes"][conc_ind] = score

                    # Attribute concepts
                    att_logits = {
                        conc_ind: (
                            det_data["conc_pos_logits"][i],
                            det_data["conc_neg_logits"][i]
                        )
                        for conc_ind, i in label_concepts_per_catType["att"].items()
                    }
                    for conc_ind, (p_logit, n_logit) in att_logits.items():
                        score = F.softmax(torch.tensor([p_logit, n_logit]))[0].item()
                        obj["pred_attributes"][conc_ind] = score

                    # Relation concepts (Within new detections)
                    for oj, det_data2 in zip(new_objs, best_detections):
                        if oi==oj: continue     # Dismiss self-self object pairs
                        x1_int, y1_int, x2_int, y2_int = _box_intersection(
                            det_data["bbox"], det_data2["bbox"]
                        )

                        bbox_intersection = (x2_int - x1_int) * (y2_int - y1_int) \
                            if x2_int > x1_int and y2_int > y1_int else 0.0
                        bbox2_A = (det_data2["bbox"][2] - det_data2["bbox"][0]) * \
                            (det_data2["bbox"][3] - det_data2["bbox"][1])

                        obj["pred_relations"][oj][0] = bbox_intersection / bbox2_A
                    
                    # Relation concepts (Between existing detections)
                    for oj in existing_objs:
                        oj_bbox = self.scene[oj]["pred_boxes"]
                        x1_int, y1_int, x2_int, y2_int = _box_intersection(
                            det_data["bbox"], oj_bbox
                        )

                        bbox_intersection = (x2_int - x1_int) * (y2_int - y1_int) \
                            if x2_int > x1_int and y2_int > y1_int else 0.0
                        bbox1_A = (det_data["bbox"][2] - det_data["bbox"][0]) * \
                            (det_data["bbox"][3] - det_data["bbox"][1])
                        bbox2_A = (oj_bbox[2] - oj_bbox[0]) * (oj_bbox[3] - oj_bbox[1])

                        obj["pred_relations"][oj][0] = bbox_intersection / bbox2_A
                        self.scene[oj]["pred_relations"][oi][0] = bbox_intersection / bbox1_A

                self.f_vecs.update({
                    oi: output.class_embeds[0][i].cpu().numpy()
                    for oi, i in zip(new_objs, best_indices)
                })

        if visualize:
            if lexicon is not None:
                lexicon = {
                    cat_type: {
                        ci: lexicon.d2s[(ci, cat_type)][0][0].split("/")[0]
                        for ci in range(getattr(self.inventories, cat_type))
                    }
                    for cat_type in ["cls", "att"]
                }
            self.summ = visualize_sg_predictions(self.last_input, self.scene, lexicon)

    def cache_vectors(self):
        """
        Pre-compute and cache feature vector outputs from the DETR model to speed up
        the training process...
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.detr.eval()

        dataset_path = self.cfg.vision.data.path
        images_path = os.path.join(dataset_path, "images")
        vectors_path = os.path.join(dataset_path, "vectors")
        os.makedirs(vectors_path, exist_ok=True)

        with open(f"{dataset_path}/annotations.json") as ann_f:
            annotations = json.load(ann_f)

        for img in tqdm.tqdm(annotations, total=len(annotations)):
            vec_path = os.path.join(vectors_path, f"{img['file_name']}.vectors")

            if len(img["annotations"]) == 0: continue
            if os.path.exists(vec_path): continue

            image_raw = os.path.join(images_path, img["file_name"])
            image_raw = Image.open(image_raw)
            if image_raw.mode != "RGB":
                # Cast non-RGB images (e.g. grayscale) into RGB format
                old_image_raw = image_raw
                image_raw = Image.new("RGB", old_image_raw.size)
                image_raw.paste(old_image_raw)

            bboxes = [obj["bbox"] for obj in img["annotations"].values()]
            bboxes = torch.tensor(bboxes).to(device)
            bboxes = box_convert(bboxes, "xywh", "cxcywh")
            bboxes = torch.stack([
                bboxes[:,0] / image_raw.width, bboxes[:,1] / image_raw.height,
                bboxes[:,2] / image_raw.width, bboxes[:,3] / image_raw.height,
            ], dim=-1)

            fvecs = self.model.fvecs_from_image_and_bboxes(image_raw, bboxes).cpu()[0]
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
            check_val_every_n_epoch=None,       # Iteration-based val
            val_check_interval=2000,
            num_sanity_val_steps=0,
            log_every_n_steps=500,
            logger=wb_logger,
            callbacks=[
                ModelCheckpoint(monitor="val_loss", save_last=True),
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

    def post_process(self, outputs, target_sizes):
        """
        Basically identical to OwlViTProcessor().post_process, except it is ensured
        all intermediate tensors are on the same device.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = _center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(device=boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results
    
    def add_concept(self, cat_type):
        """
        Register a novel visual concept to the model, expanding the concept inventory of
        corresponding category type (class/attribute/relation). Note that visual concepts
        are not inseparably attached to some linguistic symbols; such connections are rather
        incidental and should be established independently (consider synonyms, homonyms).
        Plus, this should allow more flexibility for, say, multilingual agents, though there
        is no plan to address that for now...

        Returns the index of the newly added concept.
        """
        C = getattr(self.inventories, cat_type)
        setattr(self.inventories, cat_type, C+1)
        return C


class VisualConceptInventory:
    def __init__(self):
        self.cls = self.att = self.rel = 0


def _center_to_corners_format(x):
    """
    Helper method in OwlViTFeatureExtractor module exposed for access
    """
    x_center, y_center, width, height = x.unbind(-1)
    boxes = [(x_center - 0.5 * width), (y_center - 0.5 * height), (x_center + 0.5 * width), (y_center + 0.5 * height)]
    return torch.stack(boxes, dim=-1)

def _box_intersection(box1, box2):
    """ Helper method for obtaining intersection of two boxes (xyxy format) """
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    return x1_int, y1_int, x2_int, y2_int
