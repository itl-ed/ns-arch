import os
import copy
import json
import random
import logging
from PIL import Image
from collections import defaultdict

import nltk
import torch
import networkx as nx
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

        # If populated, can be used to pre-compute and cache feature vectors for each
        # object in images
        self.fvec_extract_fn = None

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
                download_images(imgs_to_download)
                reformat_annotations(dataset_path, num_images_used, img_anns)

            # Finally, cleanup by removing files that are not needed anymore
            for f in ["image_data", "attribute_synsets", "scene_graphs"]:
                if os.path.exists(f"{dataset_path}/{f}.json"):
                    os.remove(f"{dataset_path}/{f}.json")

    def setup(self, stage):
        # Construct _SGGDataset instance with specified dataset
        dataset_path = self.cfg.vision.data.path

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
                    ann_train, dataset_path, metadata[conc_type]
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
                    ann_val, dataset_path, metadata[conc_type]
                )
                self.samplers[conc_type]["val"] = _FewShotSGGDataSampler(
                    index_val, hypernym_info,
                    self.cfg.vision.data.batch_size_eval,
                    self.cfg.vision.data.num_exs_per_conc_eval,
                    with_replacement=False
                )
            if stage in ["fit", "test", "predict"]:
                # Test set required for "fit"/"test"/"predict" stage setup
                concepts_test = concepts[split2:]
                ann_test, index_test = _annotations_by_concept(
                    annotations, conc_type, concepts_test
                )
                self.datasets[conc_type]["test"] = _SGGDataset(
                    ann_test, dataset_path, metadata[conc_type]
                )
                self.samplers[conc_type]["test"] = _FewShotSGGDataSampler(
                    index_test, hypernym_info,
                    self.cfg.vision.data.batch_size_eval,
                    self.cfg.vision.data.num_exs_per_conc_eval,
                    with_replacement=False
                )

    @staticmethod
    def _collate_fn(data):
        """ Custom collate_fn to pass to dataloaders """
        # Enforce same data type in batch
        assert len(data[0])==2
        assert all(type(d[0])==type(data[0][0]) for d in data)

        batch_data, concept_labels = list(zip(*data))

        if type(data[0][0]) == tuple:
            # Vectors not cached, raw data
            return tuple(zip(*batch_data)), concept_labels
        else:
            # Cached vectors fetched as torch tensor
            return torch.stack(batch_data, dim=0), concept_labels

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.cfg.vision.task.conc_type]["train"],
            batch_sampler=self.samplers[self.cfg.vision.task.conc_type]["train"],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.cfg.vision.task.conc_type]["val"],
            batch_sampler=self.samplers[self.cfg.vision.task.conc_type]["val"],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.cfg.vision.task.conc_type]["test"],
            batch_sampler=self.samplers[self.cfg.vision.task.conc_type]["test"],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn
        )


class _SGGDataset(Dataset):
    def __init__(self, annotations, dataset_path, concept_names):
        super().__init__()
        self.annotations = annotations
        self.dataset_path = dataset_path
        self.concept_names = concept_names
    
    def __getitem__(self, idx):
        assert len(idx) == 3
        img_id, obj_ids, conc = idx

        images_path = os.path.join(self.dataset_path, "images")
        vectors_path = os.path.join(self.dataset_path, "vectors")

        img_file = self.annotations[img_id]["file_name"]
        vecs_file = f"{img_file}.vectors"

        if os.path.exists(os.path.join(vectors_path, vecs_file)):
            # Fetch and return pre-computed feature vectors for the image
            vecs = torch.load(os.path.join(vectors_path, vecs_file))
            vecs = torch.stack([vecs[oid] for oid in obj_ids], dim=0)

            return vecs, self.concept_names[conc]
        else:
            # Fetch and return raw image, bboxes and object indices
            image_raw = os.path.join(images_path, img_file)
            image_raw = Image.open(image_raw)
            if image_raw.mode != "RGB":
                # Cast non-RGB images (e.g. grayscale) into RGB format
                old_image_raw = image_raw
                image_raw = Image.new("RGB", old_image_raw.size)
                image_raw.paste(old_image_raw)

            oids = list(self.annotations[img_id]["annotations"])
            bboxes = [
                obj["bbox"] for obj in self.annotations[img_id]["annotations"].values()
            ]
            bb_inds = [oids.index(oid) for oid in obj_ids]

            return (image_raw, bboxes, bb_inds), self.concept_names[conc]
    
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
        self, index_conc, hypernym_info, batch_size, num_exs_per_conc,
        with_replacement=True
    ):
        self.index_conc = index_conc
        self.hypernym_info = hypernym_info
        self.batch_size = batch_size
        self.num_exs_per_conc = num_exs_per_conc
        self.num_concepts = self.batch_size // self.num_exs_per_conc

        self.with_replacement = with_replacement

        # Let's keep things simple by making batch size divisible by number of
        # exemplars per concept (equivalently, number of concepts or 'ways')
        assert self.batch_size % self.num_exs_per_conc == 0

        # Build ontology tree (forest) from contained concepts and provided hypernym
        # metadata info, find connected components; later we will sample components
        # first and then sample concepts from the components, so as to avoid sampling
        # from concepts belonging to connected ontology tree
        self.ont_forest = nx.Graph()
        for c in self.index_conc:
            if len(self.index_conc[c]) >= self.num_exs_per_conc:
                self.ont_forest.add_node(c)
                if c in self.hypernym_info:
                    for h in self.hypernym_info[c]:
                        if h in self.index_conc and \
                            len(self.index_conc[h]) >= self.num_exs_per_conc:
                            self.ont_forest.add_edge(c, h)
        self.ont_forest = [
            list(comp) for comp in nx.connected_components(self.ont_forest)
        ]

    def __iter__(self):
        # For maintaining lists of concepts & exemplars as sampling candidates
        conc_ontology = copy.deepcopy(self.ont_forest)
        conc_exemplars = copy.deepcopy(self.index_conc)

        while True:
            if len(conc_ontology) < self.num_concepts:
                # Exhausted, cannot sample anymore
                return

            # First sample connected components in ontology, so as to avoid sampling
            # from more than one from same comp
            sampled_comps = random.sample(conc_ontology, self.num_concepts)

            # Then sample one concept from each sampled component
            sampled_concs = [random.sample(comp, 1)[0] for comp in sampled_comps]

            # Leave only those concepts with enough number of instances remaining
            sampled_concs = [
                conc for conc in sampled_concs
                if len(conc_exemplars[conc]) >= self.num_exs_per_conc
            ]

            # Now sample K instances per sampled concepts
            sampled_indices = [
                (random.sample(conc_exemplars[conc], self.num_exs_per_conc), conc)
                for conc in sampled_concs
            ]
            sampled_indices = [
                ind+(conc,) for inds, conc in sampled_indices for ind in inds
            ]           # Flatten and attach concept labels

            if not self.with_replacement:
                # Pop instances from exemplar lists
                concepts_to_del = set()
                for img_id, obj_ids, conc in sampled_indices:
                    conc_exemplars[conc].remove((img_id, obj_ids))

                    if len(conc_exemplars[conc]) < self.num_exs_per_conc:
                        # If number of instances drops below K, remove concept
                        # from sample candidates
                        concepts_to_del.add(conc)

                # Pop concepts to remove from ontology
                for conc in concepts_to_del:
                    for comp in conc_ontology:
                        if conc in comp: comp.remove(conc)
                    del conc_exemplars[conc]        # Not necessary but my OCD
                conc_ontology = [comp for comp in conc_ontology if len(comp) > 0]

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
