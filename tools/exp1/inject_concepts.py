"""
Script for injecting knowledge about intermediary concepts (parts and attributes)
as sets of positive/negative exemplars (feature vectors thereof), so that the
agent can perform good-enough detection of their instances
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import random
from PIL import Image
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import tqdm
import torch

import hydra
import numpy as np
from omegaconf import OmegaConf

from itl import ITLAgent
from tools.sim_user import SimulatedTeacher


TAB = "\t"

@hydra.main(config_path="../../itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Set up agent & user
    agent = ITLAgent(cfg)
    user = SimulatedTeacher(cfg, target_concepts={})
    # Also control PYTHONHASHSEED environment var (e.g. in bash script, or launch.json if in vscode)

    if torch.cuda.is_available():
        agent.vision.model.to("cuda")

    # Set of intermediary concepts to be injected -- parts, attributes, relations
    intermediary_concepts = {
        "cls": {        # Hand-picked; object parts
            "handle.n.01", "body.n.08", "meat.n.01", "tine.n.01", "foot.n.03", "label.n.01",
            "bowl.n.01", "stem.n.03", "neck.n.04", "blade.n.09", "bone.n.01"
        },
        # "cls": {        # Hand-picked; glasses and glass parts
        #     "drinking_glass.n.01",
        #     "bordeaux_glass.n.*",
        #     "brandy_glass.n.*",
        #     "burgundy_glass.n.*",
        #     "cabernet_sauvignon_glass.n.*",
        #     "champagne_coupe.n.*",
        #     "champagne_flute.n.*",
        #     "martini_glass.n.*",
        #     "pinot_noir_glass.n.*",
        #     "riesling_glass.n.*",
        #     "rose_glass.n.*",
        #     "sauternes_glass.n.*",
        #     "sauvignon_blanc_glass.n.*",
        #     "sherry_glass.n.*",
        #     "champagne_tulip.n.*",
        #     "zinfandel_glass.n.*",
        #     "bowl.n.01", "stem.n.03", "foot.n.03"
        # },
        # "cls": set(user.exemplars_cls),
        "att": set(user.exemplars_att),
        "rel": set(user.exemplars_rel)
    }

    source_imgs = []
    cls_f_vecs_all = []
    cls_pos_exs = defaultdict(set)
    pointers_cls_src = {}

    att_f_vecs_all = []
    att_pos_exs_subcat = defaultdict(lambda: defaultdict(set))
    att_neg_exs_subcat = defaultdict(lambda: defaultdict(set))
    pointers_att_src = {}

    # Class-attribute cooccurrence info
    cls_att_pairs = [
        tuple(int(i) for i in ci_ai.split("_"))
        for ci_ai in user.metadata["classes_attributes_pair_instances"]
    ]
    cls_att_pairs = [
        (user.metadata["classes_names"][ci], user.metadata["attributes_names"][ai])
        for ci, ai in cls_att_pairs
    ]

    agent.vision.model.eval()
    with torch.no_grad():
        for img in tqdm.tqdm(user.data_annotation, total=len(user.data_annotation)):
            image_path = os.path.join(user.image_dir_prefix, img["file_name"])
            image_raw = Image.open(image_path)
            cls_fv_ind_offset = len(cls_f_vecs_all)
            att_fv_ind_offset = len(att_f_vecs_all)
            cls_pos_exs_img = defaultdict(set)
            att_pos_exs_subcat_img = defaultdict(lambda: defaultdict(set))
            att_neg_exs_subcat_img = defaultdict(lambda: defaultdict(set))

            # Mapping from original object indexing to new indexing due to filtering
            # out objects of non-interest
            ind_map_o2c = { int(oi): i for i, oi in enumerate(img["annotations"]) }
            ind_map_c2o = { v: k for k, v in ind_map_o2c.items() }

            # Process the raw image + bboxes with vision module to obtain feature vectors
            bboxes = torch.tensor(
                [obj["bbox"] for obj in img["annotations"].values()]
            ).to(agent.vision.model.device)
            cls_embeddings, att_embeddings, _ = agent.vision.model(image_raw, bboxes)

            vis_out = {
                int(oi): (
                    cls_embeddings[i].cpu().numpy(),
                    att_embeddings[i].cpu().numpy()
                )
                for i, oi in enumerate(img["annotations"])
            }

            ## Process class concept exemplars

            # Storing positive class exemplar pointers as row indices of the feature
            # vector matrix
            for conc in intermediary_concepts["cls"]:
                if img["image_id"] in user.exemplars_cls[conc]:
                    objs = user.exemplars_cls[conc][img["image_id"]]
                    cls_pos_exs_img[conc] |= {
                        ind_map_o2c[oi] for oi in objs
                    }

            # All positive exemplars of interest to store
            if len(cls_pos_exs_img) > 0:
                cls_pos_exs_img_all = sorted(set.union(*cls_pos_exs_img.values()))
            else:
                cls_pos_exs_img_all = []

            # Store embeddings for exemplars, along with pointer info
            for i in range(len(cls_pos_exs_img_all)):
                pointers_cls_src[i+cls_fv_ind_offset] = (len(source_imgs), (i,))
            cls_f_vecs_all += [
                vis_out[ind_map_c2o[cls_fv_i]][0] for cls_fv_i in cls_pos_exs_img_all
            ]

            # Accumulate pos exs pointer
            for conc, cls_fv_inds in cls_pos_exs_img.items():
                cls_pos_exs[conc] |= {
                    cls_pos_exs_img_all.index(cls_fv_i)+cls_fv_ind_offset
                    for cls_fv_i in cls_fv_inds
                }

            ## Process attribute concept exemplars

            # Storing positive/negative attribute exemplars as row indices of feature
            # vector matrix
            for conc in intermediary_concepts["att"]:
                for cls_conc in cls_pos_exs_img:
                    # Semantics of attributes can be strongly affected by the class
                    # identities of the referents, so subcategorize attribute concepts
                    # accordingly; store negative exemplars with matching class concepts!
                    if (cls_conc, conc) in cls_att_pairs:
                        cls_insts = user.exemplars_cls[cls_conc][img["image_id"]]

                        pos_exs = set(user.exemplars_att[conc][img["image_id"]]) \
                            if img["image_id"] in user.exemplars_att[conc] else set()
                        neg_exs = set(cls_insts) - pos_exs

                        att_pos_exs_subcat_img[conc][cls_conc] |= {
                            ind_map_o2c[oi] for oi in pos_exs
                        }
                        att_neg_exs_subcat_img[conc][cls_conc] |= {
                            ind_map_o2c[oi] for oi in neg_exs
                        }

            # All positive/negative exemplars of interest to store
            if len(att_pos_exs_subcat_img) > 0:
                att_pos_exs_img_all = set.union(*[
                    set.union(*per_cls.values())
                    for per_cls in att_pos_exs_subcat_img.values()
                ])
            else:
                att_pos_exs_img_all = set()
            if len(att_neg_exs_subcat_img) > 0:
                att_neg_exs_img_all = set.union(*[
                    set.union(*per_cls.values())
                    for per_cls in att_neg_exs_subcat_img.values()
                ])
            else:
                att_neg_exs_img_all = set()
            att_exs_img_all = att_pos_exs_img_all | att_neg_exs_img_all

            # Need another index map from the larger class exemplar index set to the
            # likely smaller attribute exemplar index set
            ind_map_c2a = { ci: i for i, ci in enumerate(att_exs_img_all) }

            # Store embeddings for exemplars, along with pointer info
            for cls_fv_i, att_fv_i in ind_map_c2a.items():
                pointers_att_src[att_fv_i+att_fv_ind_offset] = (
                    len(source_imgs), (cls_pos_exs_img_all.index(cls_fv_i),)
                )
            att_f_vecs_all += [
                vis_out[ind_map_c2o[cls_fv_i]][1] for cls_fv_i in ind_map_c2a
            ]

            # Accumulate pos/neg exs pointer
            for conc, per_cls in att_pos_exs_subcat_img.items():
                for cls_conc, cls_fv_inds in per_cls.items():
                    att_pos_exs_subcat[conc][cls_conc] |= {
                        ind_map_c2a[ci]+att_fv_ind_offset for ci in cls_fv_inds
                    }
            for conc, per_cls in att_neg_exs_subcat_img.items():
                for cls_conc, cls_fv_inds in per_cls.items():
                    att_neg_exs_subcat[conc][cls_conc] |= {
                        ind_map_c2a[ci]+att_fv_ind_offset for ci in cls_fv_inds
                    }

            # Source image and bounding boxes corresponding to each embedding
            source_imgs.append(
                (np.asarray(image_raw), bboxes[cls_pos_exs_img_all].cpu().numpy())
            )

    cls_f_vecs_all = np.array(cls_f_vecs_all, dtype=np.float32)
    att_f_vecs_all = np.array(att_f_vecs_all, dtype=np.float32)

    cls_pos_exs = dict(cls_pos_exs)
    att_pos_exs_subcat = { k: dict(v) for k, v in att_pos_exs_subcat.items() }
    att_neg_exs_subcat = { k: dict(v) for k, v in att_neg_exs_subcat.items() }

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
    att_pos_exs = {
        (conc, cls_conc): pos_fv_inds
        for conc, per_cls in att_pos_exs_subcat.items()
        for cls_conc, pos_fv_inds in per_cls.items()
    }
    att_neg_exs = {
        (conc, cls_conc): set(
            random.sample(list(neg_fv_inds), len(att_pos_exs[(conc, cls_conc)]))
        ) if len(neg_fv_inds) > len(att_pos_exs[(conc, cls_conc)]) else neg_fv_inds
        for conc, per_cls in att_neg_exs_subcat.items()
        for cls_conc, neg_fv_inds in per_cls.items()
    }

    # Register new attribute concepts along with their linguistic data
    pointers_att_exm = {}
    for conc, cls_conc in set(att_pos_exs) | set(att_neg_exs):
        conc_ind = agent.vision.add_concept("att")

        name, pos, _ = conc.split(".")
        name = "".join(
            tok if i==0 else tok.capitalize()       # camelCase multi-token names
            for i, tok in enumerate(name.split("_"))
        )
        name += "/" + cls_conc.split(".")[0]
        agent.lt_mem.lexicon.add((name, pos), (conc_ind, "att"))

        # Packing positive & negative exemplars into a single dict
        pointers_att_exm[conc_ind] = (
            att_pos_exs[(conc, cls_conc)], att_neg_exs[(conc, cls_conc)]
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
        agent.cfg.paths.outputs_dir, f"injected_{cfg.seed}.ckpt"
    )
    agent.save_model(ckpt_path)


if __name__ == "__main__":
    main()
