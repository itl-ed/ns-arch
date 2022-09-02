"""
Script for fine-grained grounding experiments; simulate natural interactions between
agent (learner) and user (teacher) with varying configurations
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import random
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import tqdm
import torch
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data.detection_utils import convert_image_to_rgb

from itl import ITLAgent
from itl.opts import parse_arguments
from tools.sim_user import SimulatedTeacher


TAB = "\t"

if __name__ == "__main__":
    opts = parse_arguments()

    target_concepts = {
        "cls": [
            "brandy_glass.n.*",
            "burgundy_glass.n.*",
            "champagne_coupe.n.*"
        ]
    }

    # Set up agent & user
    agent = ITLAgent(opts)
    user = SimulatedTeacher(
        target_concepts=target_concepts,
        strat_feedback=opts.exp1_strat_feedback,
        test_set_size=opts.exp1_test_set_size,
        seed=opts.exp1_random_seed
    )
    # Also control PYTHONHASHSEED environment var (e.g. launch.json if in vscode)

    intermediary_concepts = {
        "cls": set(user.exemplars_cls) - set(user.domain_knowledge),
        "att": set(user.exemplars_att),
        "rel": set(user.exemplars_rel),
    }

    agent.vision.dm.setup("test")
    agent.vision.model.eval()

    # Turn off UI pop up on predict
    agent.vis_ui_on = False

    # Inject object class concepts first, on which attribute & relation concepts depend
    source_imgs = []
    cls_f_vecs_all = []
    cls_pos_exs = defaultdict(set)

    for img in tqdm.tqdm(user.data_annotation[:10], total=len(user.data_annotation[:10])):
        # Process all images in dataset
        inp = {
            "file_name": os.path.join(user.image_dir_prefix, img["file_name"]),
            "annotations": [
                {
                    "bbox": obj["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "objectness_scores": None
                }
                for obj in img["annotations"]
            ]
        }
        inp = [agent.vision.dm.mapper_batch["test_props"](inp)]

        with torch.no_grad():
            _, f_vecs, _ = agent.vision.model.base_model.inference(inp)

        # Storing positive class exemplars as row indices of feature vector matrix
        for conc in intermediary_concepts["cls"]:
            if img["image_id"] in user.exemplars_cls[conc]:
                objs = user.exemplars_cls[conc][img["image_id"]]
                cls_pos_exs[conc] |= set(oi+len(cls_f_vecs_all) for oi in objs)

        # Source image and bounding boxes corresponding to each RoI feature vector
        # (Bboxes of RoIs for relations can be recovered from bboxes of RoI pairs)
        source_imgs.append((
            convert_image_to_rgb(inp[0]["image"].permute(1, 2, 0), "BGR"),
            BoxMode.convert(
                np.stack([obj["bbox"] for obj in img["annotations"]]),
                BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
            )
        ))

        # Append vectors to matrix
        cls_f_vecs_all += list(f_vecs[0].cpu().numpy())
    
    cls_f_vecs_all = np.array(cls_f_vecs_all, dtype=np.float32)

    # Sampling negative exemplars for class concepts
    cls_neg_exs = {
        conc: set(range(len(cls_f_vecs_all))) - fv_inds
        for conc, fv_inds in cls_pos_exs.items()
    }       # Negative exemplar sets as complements of respective positive exemplar sets
    cls_neg_exs = {
        conc: set(random.sample(list(fv_inds), len(cls_pos_exs[conc])))
        for conc, fv_inds in cls_neg_exs.items()
    }       # Sampling down to the same number as positive exemplars

    # Pointing class vectors to their source objects
    oi_offsets = np.cumsum([0] + [len(bboxes) for _, bboxes in source_imgs[:-1]])
    pointers_cls_src = {
        oi+off: (i, oi)
        for i, ((_, bboxes), off) in enumerate(zip(source_imgs, oi_offsets))
        for oi in range(len(bboxes))
    }

    # Register new class concepts along with their linguistic data
    pointers_cls_exm = {}
    for conc in cls_pos_exs:
        conc_ind = agent.vision.add_concept("cls")
        name, pos, _ = conc.split(".")
        name = "".join(
            tok if i==0 else tok.capitalize()       # camelCase multi-token names
            for i, tok in enumerate(name.split("_"))
        )
        agent.lt_mem.lexicon.add((name, pos), (conc_ind, "cls"))
        # agent.vision.predicates[cat_type].append(
        #     f"{name}.{pos}.0{len(self.lt_mem.lexicon.s2d[(name, pos)])}"
        # )

        # Packing positive & negative exemplars into a single dict
        pointers_cls_exm[conc_ind] = (cls_pos_exs[conc], cls_neg_exs[conc])

    # Finally add class exemplars
    agent.lt_mem.exemplars.add_exs(
        source_imgs,
        { "cls": cls_f_vecs_all },
        { "cls": pointers_cls_src },
        { "cls": pointers_cls_exm }
    )
    # Update the category code parameter in the vision model's predictor head using
    # the new set of exemplars
    for conc_ind in pointers_cls_exm:
        concept = (conc_ind, "cls")
        agent.vision.update_concept(
            concept, agent.lt_mem.exemplars[concept], mix_ratio=1.0
        )

    # Now inject attribute & relation concepts
    source_imgs = []
    att_f_vecs_all = []; rel_f_vecs_all = []
    att_pos_exs = defaultdict(set); rel_pos_exs = defaultdict(set)

    for img in tqdm.tqdm(user.data_annotation[:10], total=len(user.data_annotation[:10])):
        # Process all images in dataset
        inp = {
            "file_name": os.path.join(user.image_dir_prefix, img["file_name"]),
            "annotations": [
                {
                    "bbox": obj["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "objectness_scores": None
                }
                for obj in img["annotations"]
            ]
        }
        inp = [agent.vision.dm.mapper_batch["test_props"](inp)]

        with torch.no_grad():
            _, f_vecs, _ = agent.vision.model.base_model.inference(inp)

        # Storing positive attribute exemplars as row indices of feature vector matrix
        for conc in intermediary_concepts["att"]:
            if img["image_id"] in user.exemplars_att[conc]:
                objs = user.exemplars_att[conc][img["image_id"]]
                att_pos_exs[conc] |= set(oi + len(att_f_vecs_all) for oi in objs)
        # Storing positive relation exemplars as row indices of feature vector matrix
        for conc in intermediary_concepts["rel"]:
            if img["image_id"] in user.exemplars_rel[conc]:
                obj_pairs = user.exemplars_rel[conc][img["image_id"]]
                rel_pos_exs[conc] |= set(
                    len(img["annotations"])*oi + oj + len(rel_f_vecs_all)
                    for oi, oj in obj_pairs
                )

        # Source image and bounding boxes corresponding to each RoI feature vector
        # (Bboxes of RoIs for relations can be recovered from bboxes of RoI pairs)
        source_imgs.append((
            convert_image_to_rgb(inp[0]["image"].permute(1, 2, 0), "BGR"),
            BoxMode.convert(
                np.stack([obj["bbox"] for obj in img["annotations"]]),
                BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
            )
        ))

        # Append vectors to matrix
        att_f_vecs_all += list(f_vecs[1].cpu().numpy())
        rel_f_vecs_all += list(f_vecs[2].view(-1, f_vecs[2].shape[-1]).cpu().numpy())
    
    att_f_vecs_all = np.array(att_f_vecs_all, dtype=np.float32)
    rel_f_vecs_all = np.array(rel_f_vecs_all, dtype=np.float32)

    # Sampling negative exemplars for attribute concepts
    att_neg_exs = {
        conc: set(range(len(att_f_vecs_all))) - fv_inds
        for conc, fv_inds in att_pos_exs.items()
    }       # Negative exemplar sets as complements of respective positive exemplar sets
    att_neg_exs = {
        conc: set(random.sample(list(fv_inds), len(att_pos_exs[conc])))
        for conc, fv_inds in att_neg_exs.items()
    }       # Sampling down to the same number as positive exemplars

    # Sampling negative exemplars for relation concepts
    rel_neg_exs = {
        conc: set(range(len(rel_f_vecs_all))) - fv_inds
        for conc, fv_inds in rel_pos_exs.items()
    }       # Negative exemplar sets as complements of respective positive exemplar sets
    rel_neg_exs = {
        conc: set(random.sample(list(fv_inds), len(rel_pos_exs[conc])))
        for conc, fv_inds in rel_neg_exs.items()
    }       # Sampling down to the same number as positive exemplars

    # For space concern, trim away 'vacuous' relation vectors not referenced for any
    # concepts; it would be a huge waste to store all O(n^2) vectors (most of which
    # don't have any meaningful relations)
    fv_inds_referenced = sorted(set.union(*[
        rel_pos_exs[conc] | rel_neg_exs[conc] for conc in rel_pos_exs
    ]))
    fv_inds_referenced_inv = {fv_i: i for i, fv_i in enumerate(fv_inds_referenced)}
    rel_f_vecs_all = rel_f_vecs_all[fv_inds_referenced]
    rel_pos_exs = {
        conc: {fv_inds_referenced_inv[fv_i] for fv_i in fv_inds}
        for conc, fv_inds in rel_pos_exs.items()
    }
    rel_neg_exs = {
        conc: {fv_inds_referenced_inv[fv_i] for fv_i in fv_inds}
        for conc, fv_inds in rel_neg_exs.items()
    }

    # Pointing attribute vectors to their source objects; we can reuse pointers_cls_src
    pointers_att_src = pointers_cls_src

    # Pointing relation vectors to their source object pairs - this is bit more delicate
    oioj_offsets = np.cumsum([0] + [len(bboxes)**2 for _, bboxes in source_imgs[:-1]])
    pointers_rel_src = {
        oioj+off: (i, (oioj // len(bboxes), oioj % len(bboxes)))
        for i, ((_, bboxes), off) in enumerate(zip(source_imgs, oioj_offsets))
        for oioj in range(len(bboxes)**2)
    }
    pointers_rel_src = {
        fv_inds_referenced_inv[fv_i]: obj_pairs
        for fv_i, obj_pairs in pointers_rel_src.items()
        if fv_i in fv_inds_referenced_inv
    }

    # Register new attribute concepts along with their linguistic data
    pointers_att_exm = {}
    for conc in att_pos_exs:
        conc_ind = agent.vision.add_concept("att")
        name, pos, _ = conc.split(".")
        name = "".join(
            tok if i==0 else tok.capitalize()       # camelCase multi-token names
            for i, tok in enumerate(name.split("_"))
        )
        agent.lt_mem.lexicon.add((name, pos), (conc_ind, "att"))

        # Packing positive & negative exemplars into a single dict
        pointers_att_exm[conc_ind] = (att_pos_exs[conc], att_neg_exs[conc])

    # Register new relation concepts along with their linguistic data
    pointers_rel_exm = {}
    for conc in rel_pos_exs:
        conc_ind = agent.vision.add_concept("rel")
        name, pos, _ = conc.split(".")
        name = "".join(
            tok if i==0 else tok.capitalize()       # camelCase multi-token names
            for i, tok in enumerate(name.split("_"))
        )
        agent.lt_mem.lexicon.add((name, pos), (conc_ind, "rel"))

        # Packing positive & negative exemplars into a single dict
        pointers_rel_exm[conc_ind] = (rel_pos_exs[conc], rel_neg_exs[conc])

    # Finally add attribute & relation exemplars
    agent.lt_mem.exemplars.add_exs(
        source_imgs,
        { "att": att_f_vecs_all, "rel": rel_f_vecs_all },
        { "att": pointers_att_src, "rel": pointers_rel_src },
        { "att": pointers_att_exm, "rel": pointers_rel_exm }
    )
    # Update the category code parameter in the vision model's predictor head using
    # the new set of exemplars
    for cat_type, pointers in [("att", pointers_att_exm), ("rel", pointers_rel_exm)]:
        for conc_ind in pointers:
            concept = (conc_ind, cat_type)
            agent.vision.update_concept(
                concept, agent.lt_mem.exemplars[concept], mix_ratio=1.0
            )

    # Save model checkpoint to output dir
    ckpt_path = os.path.join(agent.opts.output_dir_path, "injected.ckpt")
    agent.save_model(ckpt_path)
