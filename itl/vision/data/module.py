import os
import copy
import json
import random
import logging
from PIL import Image
from collections import defaultdict

import nltk
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from .vg_prepare import (
    download_and_unzip,
    process_image_metadata,
    download_images,
    reformat_annotations
)

logger = logging.getLogger(__name__)


class FewShotSGGDataModule(pl.LightningDataModule):
    """
    DataModule for preparing & loading data for few-shot scene graph generation
    task (object detection & class/attribute classification; relation classification
    may also be added in some future?). Responsible for downloading, pre-processing
    data, managing train/val/test split, and shipping appropriate dataloaders.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Create data directory at the specified path if not exists
        dataset_path = self.cfg.vision.data.path
        images_path = os.path.join(dataset_path, "images")
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)

        dataset_name = self.cfg.vision.data.name
        if dataset_name == "visual_genome":
            num_images = self.cfg.vision.data.num_images
            num_images_used = int(num_images * self.cfg.vision.data.use_percentage)

            # Prepare NLTK wordnet resources to handle ontology info
            nltk_dir = os.path.join(self.cfg.paths.assets_dir, "nltk_data")
            nltk.data.path.append(nltk_dir)
            try:
                nltk.find("corpora/wordnet.zip", paths=[nltk_dir])
            except LookupError:
                nltk.download("wordnet", download_dir=nltk_dir)
            try:
                nltk.find("corpora/omw-1.4.zip", paths=[nltk_dir])
            except LookupError:
                nltk.download("omw-1.4", download_dir=nltk_dir)

            if os.path.exists(f"{dataset_path}/annotations.json") and \
                os.path.exists(f"{dataset_path}/metadata.json"):
                # Annotation & metadata file exists, presume we already have the
                # target data in the format we need
                logger.info(f"Required data files seem already present in {dataset_path}...")
                logger.info("If you want to download and process VG data files again, " \
                    "empty the directory and re-run the script.")
            else:
                # Download original VG annotations
                for f in ["image_data", "attribute_synsets", "scene_graphs"]:
                    json_filename = f"{f}.json"
                    jsonzip_filename = f"{json_filename}.zip"

                    if os.path.exists(f"{dataset_path}/{json_filename}"):
                        logger.info(f"{json_filename} already exists, skip download")
                    else:
                        download_and_unzip(dataset_path, jsonzip_filename)

                # Download image files, reformat & save VG annotations
                img_anns, imgs_to_download = process_image_metadata(
                    dataset_path, num_images_used
                )
                reformat_annotations(dataset_path, num_images_used, img_anns)
                download_images(imgs_to_download)

            # Finally, cleanup by removing files that are not needed anymore
            for f in ["image_data", "attribute_synsets", "scene_graphs"]:
                if os.path.exists(f"{dataset_path}/{f}.json"):
                    os.remove(f"{dataset_path}/{f}.json")

    def setup(self, stage):
        # Construct _SGGDataset instance with specified dataset
        dataset_path = self.cfg.vision.data.path
        images_path = os.path.join(dataset_path, "images")

        with open(f"{dataset_path}/annotations.json") as ann_f:
            annotations = json.load(ann_f)
        with open(f"{dataset_path}/metadata.json") as meta_f:
            metadata = json.load(meta_f)

        # Train/val/test split of dataset by *concepts* (not by images) for few-shot
        # training
        self.datasets = {}; self.samplers = {}
        for conc_type in ["classes", "attributes", "relations"]:
            self.datasets[conc_type] = {}
            self.samplers[conc_type] = {}

            split1 = int(len(metadata[conc_type]) * 0.8)
            split2 = int(len(metadata[conc_type]) * 0.9)

            concepts = torch.randperm(len(metadata[conc_type]))
            hypernym_info = metadata[f"{conc_type}_hypernyms"] \
                if f"{conc_type}_hypernyms" in metadata else {}
            hypernym_info = { int(k): set(v) for k, v in hypernym_info.items() }

            if stage in ["fit"]:
                # Training set required for "fit" stage setup
                concepts_train = concepts[:split1]
                ann_train, index_train = _annotations_by_concept(
                    annotations, conc_type, concepts_train
                )
                self.datasets[conc_type]["train"] = _SGGDataset(
                    ann_train, images_path
                )
                self.samplers[conc_type]["train"] = _FewShotSGGDataSampler(
                    index_train, hypernym_info,
                    self.cfg.vision.data.batch_size,
                    self.cfg.vision.data.num_exs_per_conc
                )
            if stage in ["fit", "validate"]:
                # Validation set required for "fit"/"validate" stage setup
                concepts_val = concepts[split1:split2]
                ann_val, index_val = _annotations_by_concept(
                    annotations, conc_type, concepts_val
                )
                self.datasets[conc_type]["val"] = _SGGDataset(
                    ann_val, images_path
                )
                self.samplers[conc_type]["val"] = _FewShotSGGDataSampler(
                    index_val, hypernym_info,
                    self.cfg.vision.data.batch_size,
                    self.cfg.vision.data.num_exs_per_conc
                )
            if stage in ["fit", "test", "predict"]:
                # Test set required for "fit"/"test"/"predict" stage setup
                concepts_test = concepts[split2:]
                ann_test, index_test = _annotations_by_concept(
                    annotations, conc_type, concepts_test
                )
                self.datasets[conc_type]["test"] = _SGGDataset(
                    ann_test, images_path
                )
                self.samplers[conc_type]["test"] = _FewShotSGGDataSampler(
                    index_test, hypernym_info,
                    self.cfg.vision.data.batch_size,
                    self.cfg.vision.data.num_exs_per_conc
                )

    @staticmethod
    def _collate_fn(data):
        """ Custom collate_fn to pass to dataloaders """
        imgs = [img for img, _ in data]
        bboxes = tuple(
            [bbox_tuple[i] for _, bbox_tuple in data]
            for i in range(len(data[0][1]))
        )
        return imgs, bboxes

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.cfg.vision.task]["train"],
            batch_sampler=self.samplers[self.cfg.vision.task]["train"],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.cfg.vision.task]["val"],
            batch_sampler=self.samplers[self.cfg.vision.task]["val"],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.cfg.vision.task]["test"],
            batch_sampler=self.samplers[self.cfg.vision.task]["test"],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn
        )


class _SGGDataset(Dataset):
    def __init__(self, annotations, images_path):
        super().__init__()
        self.annotations = annotations
        self.images_path = images_path
    
    def __getitem__(self, idx):
        assert len(idx) == 2
        img_id, obj_ids = idx

        image = os.path.join(self.images_path, self.annotations[img_id]["file_name"])
        image = Image.open(image)
        bboxes = tuple(
            self.annotations[img_id]["annotations"][oid]["bbox"] for oid in obj_ids
        )

        return image, bboxes
    
    def __len__(self):
        return len(self.annotations)


class _FewShotSGGDataSampler:
    """
    Prepare batch sampler that returns few-shot 'episodes' consisting of K
    instances of N concepts (similar to typical N-way K-shot episodes for few-shot
    learning, but without support-query set distinction); each instance is passed
    as a pair of index, namely (image id, object (pair) id)
    """
    def __init__(
        self, index_conc, hypernym_info, batch_size, num_exs_per_conc
    ):
        self.index_conc = index_conc
        self.hypernym_info = hypernym_info
        self.batch_size = batch_size
        self.num_exs_per_conc = num_exs_per_conc
        self.num_concepts = self.batch_size // self.num_exs_per_conc

        # Let's keep things simple by making batch size divisible by number of
        # exemplars per concept (equivalently, number of concepts or 'ways')
        assert self.batch_size % self.num_exs_per_conc == 0

    def __iter__(self):
        while True:
            sampled_indices = []        # To yield

            # First sample N concepts one by one, ensuring none in the sampled
            # concepts is a hypernym of another
            sampled_concepts = set()
            sample_cands = {
                c for c, exs in self.index_conc.items()
                if len(exs) >= self.num_exs_per_conc
            }

            # For maintaining lists of exemplars without concept overlap
            conc_exemplars = copy.deepcopy(self.index_conc)

            while len(sampled_concepts) < self.num_concepts:
                if len(sample_cands) == 0:
                    # Cannot sample from this sampler
                    raise StopIteration

                conc = random.sample(list(sample_cands), 1)[0]

                # Add to set of sampled concepts
                sampled_concepts.add(conc)

                # Register sampled concept and hypernyms if any
                sample_cands -= {conc}
                if conc in self.hypernym_info:
                    sample_cands -= self.hypernym_info[conc]

                # Sample K exemplars from the exemplar list for the sampled concept
                sampled_indices += random.sample(
                    conc_exemplars[conc], self.num_exs_per_conc
                )

                # Exclude the sampled indices altogether to avoid treating instances
                # of the same concept as 'negative' pairs
                to_del = set()
                for conc, exs in conc_exemplars.items():
                    conc_exemplars[conc] = [
                        ind for ind in exs if ind not in sampled_indices
                    ]

                    # Remove concept from list if # of unprocessed exemplars are fewer than
                    # self.num_exs_per_conc
                    if len(conc_exemplars[conc]) < self.num_exs_per_conc:
                        to_del.add(conc)
                
                for conc in to_del:
                    del conc_exemplars[conc]

            yield sampled_indices

    def __len__(self):
        raise NotImplementedError


def _annotations_by_concept(annotations, conc_type, concepts):
    """
    Subroutine for filtering annotated images by specified concepts; returns
    filtered list of images & index by concepts
    """
    annotations_filtered = {}
    index_by_concept = defaultdict(list)
    for img in annotations:
        img_added = False
        for obj_id, anno in img["annotations"].items():
            if conc_type == "classes" or conc_type == "attributes":
                for ci in anno[conc_type]:
                    if ci in concepts:
                        index_by_concept[ci].append((img["image_id"], (obj_id,)))
                        img_added = True
            else:
                assert conc_type == "relations"
                for r in anno[conc_type]:
                    for ci in r["relation"]:
                        if ci in concepts:
                            index_by_concept[ci].append(
                                (img["image_id"], (obj_id, str(r["object_id"])))
                            )
                            img_added = True

        if img_added:
            annotations_filtered[img["image_id"]] = img

    return annotations_filtered, dict(index_by_concept)
