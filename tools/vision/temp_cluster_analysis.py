import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import copy
import json
import random
from itertools import compress

import torch
from tqdm import tqdm
from detectron2.structures import BoxMode
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize

from itl.vision import VisionModule
from itl.opts import parse_arguments


if __name__ == "__main__":
    opts = parse_arguments()
    vision = VisionModule(opts)

    vision.dm.setup("test")
    vision.model.eval()

    data_dir = os.path.join("datasets", "tabletop")
    with open(os.path.join(data_dir, "annotations.json")) as ann_f:
        annotations = json.load(ann_f)
    with open(os.path.join(data_dir, "metadata.json")) as md_f:
        md = json.load(md_f)

    mapper = vision.dm.mapper_batch["test_props"]

    bm = vision.model.base_model
    rh = bm.roi_heads
    meta = vision.model.meta

    resizer = Resize((32, 32))

    # Process all images in the dataset
    with torch.no_grad():
        for img in tqdm(annotations, desc="Processing imgs"):
            # Preprocessing input image
            inp = copy.deepcopy(img)
            inp["file_name"] = os.path.join(data_dir, "images", inp["file_name"])
            for obj in inp["annotations"]:
                obj["bbox_mode"] = BoxMode.XYWH_ABS

            inp = [mapper(inp)]

            # Obtain box features
            img_features = bm.backbone(bm.preprocess_image(inp).tensor)
            img_features = [img_features[f] for f in rh.box_in_features]

            proposals_objs = inp[0]["proposals"].to(bm.device)
            box_features = rh.box_pooler(img_features, [proposals_objs.proposal_boxes])
            box_features = rh.box_head(box_features)

            # Compute code
            cls_f, _, _ = rh.box_predictor(
                box_features, None, ([proposals_objs], None), return_features=True
            )
            cls_codes = meta.cls_code_gen(cls_f)
            
            # Store processing results
            img["box_features"] = cls_f
            img["code_from_box"] = cls_codes

            # Crop object bboxes (to use as labels)
            cropped_bboxes = []
            for bbox in inp[0]["proposals"].gt_boxes:
                x1, y1, x2, y2 = bbox
                x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
                cropped = inp[0]["image"][[2,1,0],y1:y2+1,x1:x2+1] / 255
                cropped = resizer(cropped)
                cropped_bboxes.append(cropped)
            img["cropped_images"] = cropped_bboxes
    
    # Collect vectors by object classes
    by_cls = [[] for _ in range(len(md["classes"]))]
    for img in tqdm(annotations, desc="Collecting results"):
        for oi, (obj, c_img) in enumerate(zip(img["annotations"], img["cropped_images"])):
            for c in obj["classes"]:
                cls_f = img["box_features"][oi]
                cls_code = img["code_from_box"][oi]
                by_cls[c].append((cls_f, cls_code, c_img))
    
    # Randomly select K exemplars to compute avg codes
    Ks = [1, 3, 5, 10, -1]
    by_cls_avg_codes = {
        K: [sum([c for _, c, _ in random.sample(vals, K)])/K for vals in by_cls]
            if K != -1
            else [sum([c for _, c, _ in vals])/len(vals) for vals in by_cls]
        for K in Ks
    }
    by_cls_avg_codes = {K: torch.stack(codes) for K, codes in by_cls_avg_codes.items()}
    by_cls_crops = torch.stack([
        random.sample(vals, 1)[0][2] for vals in by_cls
    ])      # Randomly select a thumbnail label for each class

    # Flatten object list
    by_obj_features = torch.cat([img["box_features"] for img in annotations])
    by_obj_codes = torch.cat([img["code_from_box"] for img in annotations])
    by_obj_crops = torch.stack(
        sum([img["cropped_images"] for img in annotations], [])
    )
    by_obj_class_labels = sum([
        [
            ",".join([md["classes"][c] for c in obj["classes"]])
            for obj in img["annotations"]
        ]
        for img in annotations
    ], [])

    # Filter leaving only glasses
    obj_glass_filter = [
        ("drinking_glass" in lbl) and ("cup" not in lbl)
        for lbl in by_obj_class_labels
    ]
    cls_glass_filter = [
        ("glass" in lbl) or ("champagne" in lbl)
        for lbl in md["classes"]
    ]

    writer = SummaryWriter("output/analysis")

    # Instance feature vectors
    writer.add_embedding(
        by_obj_features,
        label_img=by_obj_crops,
        metadata=list(zip(by_obj_class_labels)),
        # metadata_header=["Classes"],
        tag="Instance feature vectors"
    )
    # Instance feature vectors (glasses only)
    writer.add_embedding(
        by_obj_features[obj_glass_filter],
        label_img=by_obj_crops[obj_glass_filter],
        metadata=list(zip(compress(by_obj_class_labels, obj_glass_filter))),
        # metadata_header=["Classes"],
        tag="Instance feature vectors (glasses only)"
    )

    # Code vectors (1-shot from each instance)
    writer.add_embedding(
        by_obj_codes,
        label_img=by_obj_crops,
        metadata=list(zip(by_obj_class_labels)),
        # metadata_header=["Classes"],
        tag="Code vectors (1-shot from each instance)"
    )
    # Code vectors (1-shot from each instance; glasses only)
    writer.add_embedding(
        by_obj_codes[obj_glass_filter],
        label_img=by_obj_crops[obj_glass_filter],
        metadata=list(zip(compress(by_obj_class_labels, obj_glass_filter))),
        # metadata_header=["Classes"],
        tag="Code vectors (1-shot from each instance; glasses only)"
    )

    # Code vectors (average across whole)
    writer.add_embedding(
        by_cls_avg_codes[-1],
        label_img=by_cls_crops,
        metadata=list(zip(md["classes"])),
        # metadata_header=["Classes"],
        tag="Code vectors (average across whole)"
    )
    # Code vectors (average across whole; glasses only)
    writer.add_embedding(
        by_cls_avg_codes[-1][cls_glass_filter],
        label_img=by_cls_crops[cls_glass_filter],
        metadata=list(zip(compress(md["classes"], cls_glass_filter))),
        # metadata_header=["Classes"],
        tag="Code vectors (average across whole; glasses only)"
    )

    # Code vectors (10-shot)
    writer.add_embedding(
        by_cls_avg_codes[10],
        label_img=by_cls_crops,
        metadata=list(zip(md["classes"])),
        # metadata_header=["Classes"],
        tag="Code vectors (10-shot)"
    )
    # Code vectors (10-shot; glasses only)
    writer.add_embedding(
        by_cls_avg_codes[10][cls_glass_filter],
        label_img=by_cls_crops[cls_glass_filter],
        metadata=list(zip(compress(md["classes"], cls_glass_filter))),
        # metadata_header=["Classes"],
        tag="Code vectors (10-shot; glasses only)"
    )

    writer.close()
