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
import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import numpy as np
import tqdm as tqdm

from itl import ITLAgent
from itl.opts import parse_arguments


TAB = "\t"

if __name__ == "__main__":
    opts = parse_arguments()

    agent = ITLAgent(opts)
    exemplars = agent.lt_mem.exemplars

    from torch.utils.tensorboard import SummaryWriter

    for cat_type in ["cls", "att", "rel"]:
        if cat_type == "cls" or cat_type == "att":
            writer = SummaryWriter("output/ex_embs")

            fvecs_to_label = exemplars.storage_vec[cat_type]
            # fvecs_imgs = [
            #     exemplars.vec2img[cat_type][fv_i] for fv_i in range(len(fvecs_to_label))
            # ]
            # fvecs_imgs = [
            #     (
            #         exemplars.storage_img[ii]["image"] / 255,
            #         exemplars.storage_img[ii]["objects"][oi],
            #         exemplars.storage_img[ii]["original_width"],
            #         exemplars.storage_img[ii]["original_height"]
            #     )
            #     for ii, oi in fvecs_imgs
            # ]
            # label_imgs = []
            # for img, bbox, ow, oh in tqdm.tqdm(fvecs_imgs, total=len(fvecs_imgs)):
            #     resized = cv2.resize(img, dsize=(ow, oh))
            #     img_patch = resized[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            #     img_patch = cv2.resize(img_patch, dsize=(80,80))
            #     label_imgs.append(img_patch)

            labels_header = [
                agent.lt_mem.lexicon.d2s[(c, cat_type)][0][0]
                for c in exemplars.exemplars_pos[cat_type]
            ]

            labels = []
            for fv_i in range(len(fvecs_to_label)):
                label_row = [
                    "p" if fv_i in exemplars.exemplars_pos[cat_type][c]
                    else "n" if fv_i in exemplars.exemplars_neg[cat_type][c]
                    else "."
                    for c in exemplars.exemplars_pos[cat_type]
                ]
                labels.append(label_row)

            writer.add_embedding(
                fvecs_to_label,
                metadata=labels,
                metadata_header=labels_header,
                tag=f"Exemplar vectors ({cat_type})"
            )
            writer.close()
