from collections import defaultdict

import cv2
import numpy as np


class Exemplars:
    """
    Inventory of exemplars encountered, stored as feature vectors (output from vision
    module's feature extractor component). Primarily used for incremental few-shot
    registration of novel concepts.
    """
    TARGET_MAX = 80

    def __init__(self):
        # Storage of source image patches
        self.storage_img = []

        # Storage of vectors, and pointers to their source
        self.storage_vec = {
            "cls": np.array([], dtype=np.float32),
            "att": np.array([], dtype=np.float32),
            "rel": np.array([], dtype=np.float32)
        }
        self.vec2img = { "cls": {}, "att": {}, "rel": {} }

        # Labelling as dict from visual concept to list of pointers to vectors
        self.exemplars_pos = {
            "cls": defaultdict(set), "att": defaultdict(set), "rel": defaultdict(set)
        }
        self.exemplars_neg = {
            "cls": defaultdict(set), "att": defaultdict(set), "rel": defaultdict(set)
        }

    def __repr__(self):
        scene_desc = f"scenes={len(self.storage_img)}"
        obj_desc = f"objects={sum([len(img[1]) for img in self.storage_img])}"
        conc_desc = f"concepts={len(self.exemplars_pos['cls'])}" \
            f"/{len(self.exemplars_pos['att'])}" \
            f"/{len(self.exemplars_pos['rel'])}"
        return f"Exemplars({scene_desc}, {obj_desc}, {conc_desc})"
    
    def __getitem__(self, item):
        conc_ind, cat_type = item

        pos_exs = self.exemplars_pos[cat_type][conc_ind]
        pos_exs = self.storage_vec[cat_type][list(pos_exs)]
        neg_exs = self.exemplars_neg[cat_type][conc_ind]
        neg_exs = self.storage_vec[cat_type][list(neg_exs)]

        return { "pos": pos_exs, "neg": neg_exs }
    
    def add_exs(self, sources, f_vecs, pointers_src, pointers_exm):
        assert len(sources) > 0

        # Add source images and object bboxes; for storage size concern, resize into
        # smaller resolutions (so that max(width, height) <= 80) and record original
        # image sizes
        scaling_factors = [
            Exemplars.TARGET_MAX / max(img.shape[:2])
            for img, _ in sources
        ]
        self.storage_img += [
            {
                "image": cv2.resize(
                    img, dsize=(int(img.shape[1]*sf), int(img.shape[0]*sf))
                ),
                "original_width": img.shape[1],
                "original_height": img.shape[0],
                "objects": bboxes
            }
            for (img, bboxes), sf in zip(sources, scaling_factors)
        ]

        N_I = len(self.storage_img)

        for cat_type in ["cls", "att", "rel"]:
            if cat_type in pointers_src:
                assert cat_type in f_vecs
                N_C = len(self.storage_vec[cat_type])

                # Add to feature vector matrix
                self.storage_vec[cat_type] = np.concatenate([
                    f_vecs[cat_type],
                    self.storage_vec[cat_type].reshape(-1, f_vecs[cat_type].shape[-1])
                ])
                # Mapping from vector row index to source object in image
                for fv_id, (src_img, src_obj) in pointers_src[cat_type].items():
                    self.vec2img[cat_type][fv_id+N_C] = (src_img+N_I, src_obj)
                # Positive/negative exemplar tagging by vector row index
                for conc_ind, (exs_pos, exs_neg) in pointers_exm[cat_type].items():
                    exs_pos = {fv_id+N_C for fv_id in exs_pos}
                    exs_neg = {fv_id+N_C for fv_id in exs_neg}
                    self.exemplars_pos[cat_type][conc_ind] |= exs_pos
                    self.exemplars_neg[cat_type][conc_ind] |= exs_neg
