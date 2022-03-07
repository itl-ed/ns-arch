"""
Wrap detectron2 train/val/test dataset loaders as PyTorch Lightning LightningDataModule.
Taken partially and extended from tools/lightning_train_net.py at detectron2 main repo.
Also implements a data module for few-shot detection as an extension of the data module for
traditional batch-learned detection models
"""
import os
import json
import logging

import ijson
from detectron2.data import (
    DatasetCatalog, MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader
)
from detectron2.structures import BoxMode
from pytorch_lightning import LightningDataModule

from .mapper import VGBatchMapper, VGFewShotMapper
from .iterables import FewShotDataset, InteractiveDataset
from .prepare import (
    download_and_unzip,
    process_image_metadata,
    download_images,
    reformat_annotations
)


logger = logging.getLogger("vision.data")


def _vg_data_getter(spl):
    """
    Read from annotation & metadata json files for DatasetCatalog registration

    Args:
        spl (string): train/val/test split; any others are rejected
    
    Returns:
        List of lightweight representations of image metadata and annotations.
    """
    assert spl == "train" or spl == "val" or spl == "test"

    dataset_dicts = []  # To return

    md = MetadataCatalog.get(f"vg_{spl}")

    with open(os.path.join(md.data_dir, md.ann_path)) as anno_f:
        data = ijson.items(anno_f, "item")

        for img in data:
            record = {
                "image_id": img["image_id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"]
            }

            for obj in img["annotations"]:
                obj["bbox_mode"] = BoxMode.XYWH_ABS

            record["annotations"] = img["annotations"]

            dataset_dicts.append(record)
    
    return dataset_dicts


class VGDataModule(LightningDataModule):

    VG_TOTAL_COUNT = 108077  # Visual genome data stat

    def __init__(
        self, cfg,
        *,
        repo_url="http://visualgenome.org/static/data/dataset",
        data_dir="datasets/visual_genome",
        img_count=-1,
        train_aug_transforms=[]
    ):
        """
        Args:
            cfg: detectron2 config dict instance
            repo_url: URL of Visual Genome repository to download files from
            data_dir: Target directory where data files will be stored
            img_count: Won't download more image files than this number; mostly
                for development purpose, -1 for no limit
            train_aug_transforms: a list of augmentations or deterministic transforms
                to apply when training scene graph generation models
        """
        super().__init__()
        self.cfg = cfg

        self.repo_url = repo_url
        self.data_dir = data_dir
        self.img_count = self.VG_TOTAL_COUNT if img_count==-1 else img_count
        self.train_aug_transforms = train_aug_transforms

        # Few-shot episode loader params; None or (N, K, I) where N=ways, K=shots, I=# instances
        # in each batch (corresponding to |support|+|query|)
        self.few_shot = None

    def prepare_data(self):
        """
        1) Download and extract necessary files (metadata & scene graph annotations) from
            the Visual Genome repository (http://visualgenome.org/static/data/dataset/...)
        2) Download actual image files from the URLs included in the fetched metadata
            (esp. attributes). Make train/val/test splits along the way.
        3) Assemble & reformat the annotations into the format expected by our custom
            Detectron2 data reader, making sure predicates are properly canonicalized
        """
        # Create data directory at the specified path if not exists
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "images"), exist_ok=True)

        # If data files with these names below already exist in the data directory, consider
        # we already have the 'target' data files needed and skip steps 1) & 2)
        if os.path.exists(f"{self.data_dir}/annotations_train.json") and \
            os.path.exists(f"{self.data_dir}/annotations_val.json") and \
            os.path.exists(f"{self.data_dir}/annotations_test.json") and \
            os.path.exists(f"{self.data_dir}/metadata.json"):

            logger.warning(f"Required data files seem already present in {self.data_dir}...")
            logger.warning("If you want to download and process VG data files again, " \
                "empty the directory and re-run the script")

            with open(f"{self.data_dir}/metadata.json") as meta_f:
                metadata = json.load(meta_f)
                total_cnt = metadata["train_cnt"] + metadata["val_cnt"] + metadata["test_cnt"]

                if total_cnt != self.img_count:
                    logger.warning(f"Mismatch between existing data count in metadata ({total_cnt})" \
                        f" vs. provided img_count arg ({self.img_count})")

        else:
            # Download original VG annotations
            for f in ["image_data", "attribute_synsets", "scene_graphs"]:
                json_filename = f"{f}.json"
                jsonzip_filename = f"{json_filename}.zip"

                if os.path.exists(f"{self.data_dir}/{json_filename}"):
                    logger.info(f"{json_filename} already exists, skip download")
                else:
                    url_path = f"{self.repo_url}/{jsonzip_filename}"
                    download_and_unzip(url_path, self.data_dir, jsonzip_filename)
            
            # Download image files, reformat & save VG annotations
            img_anns, imgs_to_download = process_image_metadata(self.data_dir, self.img_count)
            download_images(imgs_to_download)
            reformat_annotations(self.data_dir, self.img_count, img_anns)
        
        # Finally, cleanup by removing files that are not needed anymore
        for f in ["image_data", "attribute_synsets", "scene_graphs"]:
            if os.path.exists(f"{self.data_dir}/{f}.json"):
                os.remove(f"{self.data_dir}/{f}.json")
    
    def setup(self, stage):
        # Data splits to process
        if stage == "fit":
            splits = ["train", "val"]
        elif stage == "validate":
            splits = ["val"]
        elif stage == "test":
            splits = ["test"]
        else:
            splits = []

        # VG dataset mappers
        if self.few_shot:
            self.mapper = {
                "train": VGFewShotMapper(
                    self.cfg, is_train=True, augmentations=self.train_aug_transforms
                ),
                "test": VGFewShotMapper(self.cfg, is_train=False)
            }
        else:
            self.mapper = {
                "train": VGBatchMapper(
                    self.cfg, is_train=True, augmentations=self.train_aug_transforms
                ),
                "test": VGBatchMapper(self.cfg, is_train=False),
                "test_props": VGBatchMapper(self.cfg, is_train=False, provide_proposals=True)
            }

        # Load and store VG metadata in MetadataCatalog
        with open(f"{self.data_dir}/metadata.json") as meta_f:
            data = json.load(meta_f)

            # MetadataCatalogs
            for spl in splits:
                md = MetadataCatalog.get(f"vg_{spl}")
                md.count = data[f"{spl}_cnt"]
                md.data_dir = self.data_dir
                md.ann_path = data[f"{spl}_ann_path"]
                md.classes = data["classes"]
                md.attributes = data["attributes"]
                md.relations = data["relations"]
                md.obj_counts = data["obj_counts"]
                md.obj_pair_counts = data["obj_pair_counts"]
                md.classes_counts = data["classes_counts"]
                md.attributes_counts = data["attributes_counts"]
                md.relations_counts = data["relations_counts"]

                num_preds = {
                    "cls": len(md.classes),
                    "att": len(md.attributes),
                    "rel": len(md.relations)
                }

                # Register VG datasets in DatasetCatalog
                DatasetCatalog.register(
                    f"vg_{spl}", lambda spl=spl: _vg_data_getter(spl)
                )

                # Number of predicates info
                if spl == "train":
                    self.mapper["train"].num_preds = num_preds
                else:
                    self.mapper["test"].num_preds = num_preds
                    if "test_props" in self.mapper:
                        self.mapper["test_props"].num_preds = num_preds

    def train_dataloader(self):
        if self.few_shot:
            loader = build_detection_train_loader(
                FewShotDataset(self.cfg.DATASETS.TRAIN[0], self.few_shot),
                mapper=self.mapper["train"],
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                total_batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                aspect_ratio_grouping=False
            )
        else:
            loader = build_detection_train_loader(
                self.cfg, mapper=self.mapper["train"],
                num_workers=self.cfg.DATALOADER.NUM_WORKERS
            )

        return loader

    def val_dataloader(self):
        """
        Source code is the same with self.test_dataloader(), but self.cfg.DATASETS.TEST
        will have been set differently
        """
        return self.test_dataloader()

    def test_dataloader(self):
        if self.few_shot:
            loaders = build_detection_test_loader(
                FewShotDataset(self.cfg.DATASETS.TEST[0], self.few_shot, is_train=False),
                mapper=self.mapper["test"],
                num_workers=self.cfg.DATALOADER.NUM_WORKERS
            )
        else:
            loaders = [
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST, mapper=self.mapper["test"],
                    num_workers=self.cfg.DATALOADER.NUM_WORKERS
                ),
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST, mapper=self.mapper["test_props"],
                    num_workers=self.cfg.DATALOADER.NUM_WORKERS
                )
            ]

        return loaders
    
    def predict_dataloader(self):
        return build_detection_test_loader(
            InteractiveDataset(os.path.join(self.data_dir, "images")),
            mapper=self.mapper["test"]
        )
