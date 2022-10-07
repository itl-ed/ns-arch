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
from itertools import product
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import tqdm
import torch

import numpy as np
from torchvision.ops import clip_boxes_to_image, box_convert, box_iou

from itl import ITLAgent
from itl.opts import parse_arguments
from tools.sim_user import SimulatedTeacher


TAB = "\t"

if __name__ == "__main__":
    opts = parse_arguments()

    # Set up agent & user
    agent = ITLAgent(opts)
    user = SimulatedTeacher(
        target_concepts={},
        strat_feedback=opts.exp1_strat_feedback,
        test_set_size=opts.exp1_test_set_size,
        seed=opts.exp1_random_seed
    )
    # Also control PYTHONHASHSEED environment var (e.g. in bash script, or launch.json if in vscode)

    # Set of intermediary concepts to be injected -- parts, attributes, relations
    intermediary_concepts = {
        "cls": {        # Hand-picked
            "handle.n.01", "body.n.08", "meat.n.01", "tine.n.01", "foot.n.03", "label.n.01",
            "bowl.n.01", "stem.n.03", "neck.n.04", "blade.n.09", "bone.n.01"
        },
        "att": set(user.exemplars_att),
        "rel": set(user.exemplars_rel)
    }

    # Text labels to feed to OwL-ViT for aiding search, each hand-prepared
    owlvit_labels = {
        ("bowl.n.01", "drinking_glass.n.01"): "bowl part of drinking glass",
        ("stem.n.03", "drinking_glass.n.01"): "stem of drinking glass",
        ("foot.n.03", "drinking_glass.n.01"): "foot of drinking glass",
        ("label.n.01", "wine_bottle.n.01"): "label of wine bottle",
        ("body.n.08", "wine_bottle.n.01"): "lower body of bottle",
        ("neck.n.04", "wine_bottle.n.01"): "neck of wine bottle",
        ("bone.n.01", "chicken.n.01"): "bone of chicken",
        ("meat.n.01", "chicken.n.01"): "meat of chicken",
        ("bowl.n.01", "spoon.n.01"): "head of utensil",
        ("tine.n.01", "fork.n.01"): "head of utensil",
        ("blade.n.09", "knife.n.01"): "blade of knife",
        ("handle.n.01", "utensil.n.01"): "handle of utensil",
        ("handle.n.01", "coffee_mug.n.01"): "handle of mug",
        ("body.n.08", "coffee_mug.n.01"): "body of mug"
    }

    source_imgs = []
    cls_f_vecs_all = []
    cls_pos_exs = defaultdict(set)
    pointers_cls_src = {}

    att_f_vecs_all = []
    att_pos_exs_subcat = defaultdict(lambda: defaultdict(set))
    pointers_att_src = {}

    with torch.no_grad():
        for img in tqdm.tqdm(user.data_annotation, total=len(user.data_annotation)):
            image_path = os.path.join(user.image_dir_prefix, img["file_name"])
            cls_fv_ind_offset = len(cls_f_vecs_all)
            att_fv_ind_offset = len(att_f_vecs_all)

            # Collect object parts to be annotated, and total set of matching label texts
            # (Not really that necessary to prepare this label text set, just for sanity
            # check purpose...)
            obj_annos = [
                (
                    [user.metadata["classes"][c] for c in obj["classes"]],
                    [
                        (
                            [user.metadata["relations"][r] for r in rs["relation"]],
                            rs["object_id"]
                        )
                        for rs in obj["relations"]
                    ],
                    obj["object_id"]
                )
                for obj in img["annotations"]
            ]
            obj_annos = [
                (
                    obj[0],
                    [img["annotations"][rs[1]] for rs in obj[1] if "of.r.01" in rs[0]],
                    obj[2]
                )
                for obj in obj_annos
            ]
            obj_annos = [
                (
                    obj[0],
                    [user.metadata["classes"][c] for c in obj[1][0]["classes"]],
                    obj[2]
                )
                for obj in obj_annos if len(obj[1])>0
            ]
            obj_annos = {
                obj[2]: [
                    owlvit_labels[part_whole] for part_whole in product(obj[0], obj[1])
                    if part_whole in owlvit_labels
                ]
                for obj in obj_annos
            }
            label_texts = sum(obj_annos.values(), [])
            label_texts = list(set(label_texts))

            # Mapping from original object indexing to new indexing due to filtering
            # out objects of non-interest
            ind_map_o2c = { oi: i for i, oi in enumerate(obj_annos) }
            ind_map_c2o = { v: k for k, v in ind_map_o2c.items() }

            # Process with OwL-ViT to obtain boxes and embeddings for image patches
            results, output, input_img, _ = agent.vision.owlvit_process(
                image_path, label_texts=label_texts
            )
            patch_boxes = clip_boxes_to_image(
                results[0]["boxes"], (input_img.height, input_img.width)
            )
            patch_embs = output.class_embeds[0]

            # IoU with ground-truth object bboxes, then find patches with max IoU values
            gt_boxes = torch.Tensor([obj["bbox"] for obj in img["annotations"]])
            gt_boxes = gt_boxes.to(device=patch_boxes.device)
            gt_boxes = box_convert(gt_boxes, "xywh", "xyxy")
            ious = box_iou(gt_boxes, patch_boxes).max(dim=1)

            # For each ground truth bbox, select the patch with max IoU
            best_patches = { oi: ious.indices[oi].item() for oi in obj_annos }
            best_patches = {
                oi: (
                    patch_boxes[patch_ind].cpu().numpy(),
                    patch_embs[patch_ind].cpu().numpy()
                )
                for oi, patch_ind in best_patches.items()
            }

            ## Process class concept exemplars

            # Storing positive class exemplar pointers as row indices of the feature
            # vector matrix
            for conc in intermediary_concepts["cls"]:
                if img["image_id"] in user.exemplars_cls[conc]:
                    objs = user.exemplars_cls[conc][img["image_id"]]
                    cls_pos_exs[conc] |= {
                        ind_map_o2c[oi]+cls_fv_ind_offset for oi in objs
                    }
            
            # Store patch embeddings for exemplars, along with pointer info
            for cls_fv_i, _ in enumerate(best_patches.values()):
                pointers_cls_src[cls_fv_i+cls_fv_ind_offset] = (len(source_imgs), (cls_fv_i,))
            cls_f_vecs_all += [emb for _, emb in best_patches.values()]
            
            ## Process attribute concept exemplars; right now we use class & attribute
            ## feature vectors in the exactly same space, but this might be relaxed if
            ## one chooses to use difference vector spaces?

            # First catalogue set of positive exemplars for all attribute concepts in
            # this image, which will be most likely a subset of equivalent exemplar set
            # for class concepts
            all_att_pos_exs = [
                user.exemplars_att[conc][img["image_id"]]
                for conc in intermediary_concepts["att"]
                if img["image_id"] in user.exemplars_att[conc]
            ]
            all_att_pos_exs = set(sum(all_att_pos_exs, []))
            all_att_pos_exs = {ind_map_o2c[oi] for oi in all_att_pos_exs}

            # Need another index map from the larger class exemplar index set to the
            # likely smaller attribute exemplar index set
            ind_map_c2a = { ci: i for i, ci in enumerate(all_att_pos_exs) }

            # Storing positive attribute exemplars as row indices of feature vector
            # matrix
            for conc in intermediary_concepts["att"]:
                if img["image_id"] in user.exemplars_att[conc]:
                    objs = user.exemplars_att[conc][img["image_id"]]
                    # Semantics of attributes can be strongly affected by the class
                    # identities of the referents, so subcategorize attribute concepts
                    # accordingly
                    for cls_conc in cls_pos_exs:
                        att_pos_exs_subcat[conc][cls_conc] |= {
                            ind_map_c2a[ind_map_o2c[oi]]+att_fv_ind_offset
                            for oi in objs
                            if img["image_id"] in user.exemplars_cls[cls_conc] and \
                                oi in user.exemplars_cls[cls_conc][img["image_id"]]
                        }

            # Store patch embeddings for exemplars, along with pointer info
            for cls_fv_i, att_fv_i in ind_map_c2a.items():
                pointers_att_src[att_fv_i+att_fv_ind_offset] = (len(source_imgs), (cls_fv_i,))
            att_f_vecs_all += [
                best_patches[ind_map_c2o[cls_fv_i]][1] for cls_fv_i in ind_map_c2a
            ]

            # Source image and bounding boxes corresponding to each embedding
            source_imgs.append((
                np.asarray(input_img),
                np.stack(bbox for bbox, _ in best_patches.values())
            ))

    cls_pos_exs = dict(cls_pos_exs)
    att_pos_exs_subcat = dict(att_pos_exs_subcat)
    cls_f_vecs_all = np.array(cls_f_vecs_all, dtype=np.float32)
    att_f_vecs_all = np.array(att_f_vecs_all, dtype=np.float32)

    # List negative exemplars for class concepts
    cls_neg_exs = {
        conc: set(range(len(cls_f_vecs_all))) - fv_inds
        for conc, fv_inds in cls_pos_exs.items()
    }       # Negative exemplar sets as complements of respective positive exemplar sets
    cls_neg_exs = {
        conc: set(random.sample(list(fv_inds), len(cls_pos_exs[conc])))
        for conc, fv_inds in cls_neg_exs.items()
    }       # Sampling down to the same number as positive exemplars

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

        # Packing positive & negative exemplars into a single dict
        pointers_cls_exm[conc_ind] = (cls_pos_exs[conc], cls_neg_exs[conc])

    # Listing positive/negative exemplars for attribute concepts
    att_all_exs = {
        (conc, agent.lt_mem.lexicon.s2d[(cls_conc.split(".")[0], "n")][0][0]): (
            pos_fv_inds, set(range(len(att_f_vecs_all))) - pos_fv_inds
        )
        for conc, per_cls in att_pos_exs_subcat.items()
        for cls_conc, pos_fv_inds in per_cls.items()
        if len(pos_fv_inds) > 0
    }
    att_pos_exs = {
        conc_subcat: pos_fv_inds
        for conc_subcat, (pos_fv_inds, _) in att_all_exs.items()
    }
    att_neg_exs = {
        conc_subcat: set(random.sample(list(neg_fv_inds), len(pos_fv_inds)))
        for conc_subcat, (pos_fv_inds, neg_fv_inds) in att_all_exs.items()
    }

    # Register new attribute concepts along with their linguistic data
    pointers_att_exm = {}
    for conc, cls_conc_ind in att_pos_exs:
        conc_ind = agent.vision.add_concept("att")
        name, pos, _ = conc.split(".")
        name = "".join(
            tok if i==0 else tok.capitalize()       # camelCase multi-token names
            for i, tok in enumerate(name.split("_"))
        )
        name += "/" + agent.lt_mem.lexicon.d2s[(cls_conc_ind, "cls")][0][0]
        agent.lt_mem.lexicon.add((name, pos), (conc_ind, "att"))

        # Packing positive & negative exemplars into a single dict
        pointers_att_exm[conc_ind] = (
            att_pos_exs[(conc, cls_conc_ind)], att_neg_exs[(conc, cls_conc_ind)]
        )

    # Don't forget to register the "have" relation concept, though it is not neurally
    # predicted at this point and no exemplars are to be added
    conc_ind = agent.vision.add_concept("rel")
    agent.lt_mem.lexicon.add(("have", "v"), (conc_ind, "rel"))

    # Finally add all the exemplars
    agent.lt_mem.exemplars.add_exs(
        sources=source_imgs,
        f_vecs={ "cls": cls_f_vecs_all, "att": att_f_vecs_all },
        pointers_src={ "cls": pointers_cls_src, "att": pointers_att_src },
        pointers_exm={ "cls": pointers_cls_exm, "att": pointers_att_exm }
    )

    # Save model checkpoint to output dir
    ckpt_path = os.path.join(
        agent.opts.output_dir_path, f"injected_{opts.exp1_random_seed}.ckpt"
    )
    agent.save_model(ckpt_path)
