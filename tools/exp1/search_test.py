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

    if torch.cuda.is_available():
        agent.vision.model.to("cuda")
    agent.vision.model.eval()

    os.makedirs(os.path.join(cfg.paths.outputs_dir, "search_test"), exist_ok=True)

    # Randomly sample 100 images...
    images_dir = os.path.join(cfg.paths.data_dir, "tabletop", "images")
    img_files = random.sample(os.listdir(images_dir), 100)
    for img_file in tqdm.tqdm(img_files, total=len(img_files)):
        image = os.path.join(images_dir, img_file)
        image = Image.open(image)

        # Obtain encoder output, then generate proposal templates
        enc_emb, _, spatial_shapes, _, mask_flatten = detr_enc_outputs(
            agent.vision.model.detr, image, agent.vision.model.feature_extractor
        )
        per_scale_split = np.cumsum([0]+[ss.prod().item() for ss in spatial_shapes])

        gen_proposals_fn = agent.vision.model.detr.model.gen_encoder_output_proposals
        _, output_proposals = gen_proposals_fn(
            enc_emb, ~mask_flatten, spatial_shapes
        )

        # Try searching for the following parts of drinking glasses; bowl, stem, foot
        for part in ["bowl", "stem", "foot"]:
            # Obtain positive exemplars and average to prototype
            conc_ind = agent.lt_mem.lexicon.s2d[(part, "n")][0][0]
            pos_exs_inds = exemplars.exemplars_pos["cls"][conc_ind]
            pos_exs_vecs = exemplars.storage_vec["cls"][list(pos_exs_inds)]
            pos_exs_vecs = torch.tensor(pos_exs_vecs).to(agent.vision.model.device)
            pos_proto = pos_exs_vecs.mean(dim=0)

            # Project to search spec embedding space
            spec_emb = torch.cat([pos_proto, torch.zeros_like(pos_proto)], dim=-1)
            spec_emb = agent.vision.model.fs_spec_fuse(spec_emb)

            embs_concat = torch.cat([
                enc_emb, spec_emb[None, None].expand_as(enc_emb)
            ], dim=-1)

            # Perform search by computing 'compatibility scores' and predicting bboxes
            # (delta thereof, w.r.t. the templates generated)
            search_scores = torch.einsum(
                "bqd,d->bq", enc_emb,
                agent.vision.model.fs_search_match(spec_emb)
            )
            search_scores = search_scores[0].sigmoid()
            search_coord_deltas = agent.vision.model.fs_search_bbox(embs_concat)
            search_coord_logits = search_coord_deltas + output_proposals
            search_coords = search_coord_logits[0].sigmoid()
            search_coords = torch.stack([
                search_coords[:,0] * image.width, search_coords[:,1] * image.height,
                search_coords[:,2] * image.width, search_coords[:,3] * image.height
            ], dim=-1)
            search_coords = box_convert(search_coords, "cxcywh", "xyxy")
            search_coords = clip_boxes_to_image(
                search_coords, (image.height, image.width)
            )

            # Top k=10 search outputs (after) nms
            k = 10; iou_thres = 0.65
            topk_inds = nms(search_coords, search_scores, iou_thres)[:k]

            # Plot result; overlay heatmap on image, then plot top-k proposals
            _, axs = plt.subplots(nrows=2, ncols=2)
            for i in range(len(per_scale_split)-1):
                r, c = i // 2, i % 2
                scores_per_level = search_scores[per_scale_split[i]:per_scale_split[i+1]]

                scores_per_level = scores_per_level.view(
                    spatial_shapes[i][0], spatial_shapes[i][1]
                )

                axs[r, c].imshow(image)
                heatmap = scores_per_level.cpu().numpy()
                heatmap = cv2.resize(
                    heatmap, dsize=(image.width, image.height),
                    interpolation=cv2.INTER_NEAREST
                )
                axs[r, c].imshow(heatmap, cmap='rainbow_alpha')

                for topk_i in topk_inds:
                    if topk_i < per_scale_split[i] or topk_i >= per_scale_split[i+1]:
                        continue

                    # Bounding box rectangle
                    x1, y1, x2, y2 = search_coords[topk_i].tolist()
                    rec = Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=1, edgecolor="r", facecolor="none"
                    )

                    # Search score
                    axs[r, c].add_patch(rec)
                    text_label = axs[r, c].text(
                        x1, y1, f"{search_scores[topk_i].item():.3f}",
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
