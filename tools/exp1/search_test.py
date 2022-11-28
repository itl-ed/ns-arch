"""
Script for inspecting the vision module's exemplar-based search capability
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import random
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

import hydra
import cv2
import torch
import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from torchvision.ops import box_convert, clip_boxes_to_image, nms
from omegaconf import OmegaConf

from itl import ITLAgent
from itl.vision.modeling.detr_abridged import detr_enc_outputs


TAB = "\t"

@hydra.main(config_path="../../itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Setting custom colormap where lower scores give lower alpha (more transparent)
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
    color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
    plt.register_cmap(cmap=map_object)

    # Set up agent
    agent = ITLAgent(cfg)
    exemplars = agent.lt_mem.exemplars

    agent.vision.model.eval()

    os.makedirs(os.path.join(cfg.paths.outputs_dir, "search_test"), exist_ok=True)

    # Randomly sample 10 images...
    images_dir = os.path.join(cfg.paths.data_dir, "tabletop", "images")
    img_files = random.sample(os.listdir(images_dir), 10)
    for img_file in tqdm.tqdm(img_files, total=len(img_files)):
        image = os.path.join(images_dir, img_file)
        image = Image.open(image)

        # Try searching for the following parts of drinking glasses; bowl, stem, foot
        for part in ["bowl", "stem", "foot"]:
            # Obtain positive exemplars and average to prototype
            conc_ind = agent.lt_mem.lexicon.s2d[(part, "n")][0][0]
            pos_exs_inds = exemplars.exemplars_pos["cls"][conc_ind]
            pos_exs_vecs = exemplars.storage_vec["cls"][list(pos_exs_inds)]
            pos_exs_vecs = torch.tensor(pos_exs_vecs).to(agent.vision.model.device)

            proposals, scores = \
                agent.vision.model.search(image, [("cls", pos_exs_vecs)], 10)

            # Plot result; overlay heatmap on image, then plot top-k proposals
            plt.imshow(image)
            ax = plt.gca()

            for bbox, score in zip(proposals, scores):
                # Bounding box rectangle
                x1, y1, x2, y2 = bbox.tolist()
                rec = Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=1, edgecolor="r", facecolor="none"
                )
                ax.add_patch(rec)

                # Search score
                text_label = ax.text(
                    x1, y1, f"{score.item():.3f}",
                    color="w", fontsize=7
                )
                text_label.set_bbox(
                    { "facecolor": "r", "alpha": 0.5, "edgecolor": "r" }
                )

            plt.savefig(os.path.join(
                cfg.paths.outputs_dir, "search_test", f"{img_file.strip('.jpg')}_{part}.png"
            ))


if __name__ == "__main__":
    main()
