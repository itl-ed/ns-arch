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
from torch.utils.data import Dataset, DataLoader, default_collate

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
                download_images(imgs_to_download)
                reformat_annotations(dataset_path, num_images_used, img_anns, self.cfg.seed)

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
        
        for_search_task = self.cfg.vision.task == "fs_search"

        # Prepare dataloaders and samplers according to train/val/test split provided
        # in metadata
        self.datasets = {}; self.samplers = {}
        for conc_type in ["classes", "attributes"]: #, "relations"]:
            self.datasets[conc_type] = {}
            self.samplers[conc_type] = {}

            hypernym_info = metadata[f"{conc_type}_hypernyms"] \
                if f"{conc_type}_hypernyms" in metadata else {}
            hypernym_info = { int(k): set(v) for k, v in hypernym_info.items() }

            if stage in ["fit"]:
                # Training set required for "fit" stage setup
                index_train, ann_train = _annotations_by_concept(
                    annotations, metadata, conc_type, "train"
                )
                self.datasets[conc_type]["train"] = _SGGDataset(
                    ann_train, dataset_path, conc_type, metadata[f"{conc_type}_names"],
                    for_search_task=for_search_task
                )
                self.samplers[conc_type]["train"] = _FewShotSGGDataSampler(
                    index_train, hypernym_info,
                    self.cfg.vision.data.batch_size,
                    self.cfg.vision.data.num_exs_per_conc,
                    with_replacement=True
                )
            if stage in ["fit", "validate"]:
                # Validation set required for "fit"/"validate" stage setup
                index_val, ann_val = _annotations_by_concept(
                    annotations, metadata, conc_type, "val"
                )
                self.datasets[conc_type]["val"] = _SGGDataset(
                    ann_val, dataset_path, conc_type, metadata[f"{conc_type}_names"],
                    for_search_task=for_search_task
                )
                self.samplers[conc_type]["val"] = _FewShotSGGDataSampler(
                    index_val, hypernym_info,
                    self.cfg.vision.data.batch_size_eval,
                    self.cfg.vision.data.num_exs_per_conc_eval,
                    for_search_task=for_search_task
                )
            if stage in ["fit", "test", "predict"]:
                # Test set required for "fit"/"test"/"predict" stage setup
                index_test, ann_test = _annotations_by_concept(
                    annotations, metadata, conc_type, "test"
                )
                self.datasets[conc_type]["test"] = _SGGDataset(
                    ann_test, dataset_path, conc_type, metadata[f"{conc_type}_names"],
                    for_search_task=for_search_task
                )
                self.samplers[conc_type]["test"] = _FewShotSGGDataSampler(
                    index_test, hypernym_info,
                    self.cfg.vision.data.batch_size_eval,
                    self.cfg.vision.data.num_exs_per_conc_eval,
                    for_search_task=for_search_task
                )

        # Also setup class+attribute hybrid dataloader for few-shot search task
        if for_search_task:
            self.datasets["classes_and_attributes"] = {}
            self.samplers["classes_and_attributes"] = {}

            splits = set(self.datasets["classes"])
            assert set(self.datasets["classes"]) == set(self.datasets["attributes"])

            # Small helper class for handling getting concept name tuples
            class _ConcNameGetter:
                def __init__(self, names):
                    self.names = names
                def __getitem__(self, idxs):
                    assert len(idxs) == len(self.names)
                    return tuple(self.names[ni][idx] for ni, idx in enumerate(idxs))

            cls_att_pairs = {
                tuple(int(i) for i in ci_ai.split("_")): [
                    (int(img_id), (oi,))
                    for img_id, obj_ids in insts.items() for oi in obj_ids
                ]
                for ci_ai, insts in metadata["classes_attributes_pair_instances"].items()
            }

            for spl in splits:
                # Merged _SGGDataset
                cls_data = self.datasets["classes"][spl]
                att_data = self.datasets["attributes"][spl]

                self.datasets["classes_and_attributes"][spl] = _SGGDataset(
                    { **cls_data.annotations, **att_data.annotations },
                    dataset_path, ("classes", "attributes"),
                    _ConcNameGetter([cls_data.conc_names, att_data.conc_names]),
                    for_search_task=True
                )

                # Batch sampler for the merged _SGGDataset
                cls_sampler = self.samplers["classes"][spl]
                att_sampler = self.samplers["attributes"][spl]

                # Join index_conc to form class+attribute combination entries
                combi_index_conc = {}
                K = cls_sampler.num_exs_per_conc
                for (ci, ai), insts in cls_att_pairs.items():
                    if ci not in cls_sampler.index_conc: continue
                    if ai not in att_sampler.index_conc: continue
                    if len(insts) < K: continue

                    combi_index_conc[(ci, ai)] = insts

                # Compose hypernym info; among cls+att combinations present, if either
                # cls or att is hypernym of respective type, add as hypernym
                combi_hypernym_info = {
                    (ci1, ai1): {
                        (ci2, ai2) for ci2, ai2 in combi_index_conc
                        if (ci1 in cls_sampler.hypernym_info and \
                            ci2 in cls_sampler.hypernym_info[ci1]) or \
                            (ai1 in att_sampler.hypernym_info and \
                            ai2 in att_sampler.hypernym_info[ai1])
                    }
                    for ci1, ai1 in combi_index_conc
                }
                combi_hypernym_info = {
                    (ci, ai): hypernyms
                    for (ci, ai), hypernyms in combi_hypernym_info.items()
                    if len(hypernyms) > 0
                }

                self.samplers["classes_and_attributes"][spl] = _FewShotSGGDataSampler(
                    combi_index_conc, combi_hypernym_info,
                    cls_sampler.batch_size, cls_sampler.num_exs_per_conc,
                    with_replacement=cls_sampler.with_replacement
                )

    def train_dataloader(self):
        return self._return_dataloaders("train")
    
    def val_dataloader(self):
        return self._return_dataloaders("val")

    def test_dataloader(self):
        return self._return_dataloaders("test")

    def _return_dataloaders(self, split):
        if self.cfg.vision.task == "fs_classify":
            # For few-shot classification task, return single dataloader for the
            # corresponding concept type and split
            return [
                self._fetch_dataloader("classes", split),
                self._fetch_dataloader("attributes", split)
            ]
        elif self.cfg.vision.task == "fs_search":
            # For few-shot search task, return list of dataloaders for the split,
            # class-only loader, attribute-only loader and class+attribute loader
            return [
                self._fetch_dataloader("classes", split),
                self._fetch_dataloader("attributes", split),
                self._fetch_dataloader("classes_and_attributes", split)
            ]
        else:
            raise ValueError("Invalid prediction task type")

    def _fetch_dataloader(self, conc_type, split):
        return DataLoader(
            dataset=self.datasets[conc_type][split],
            batch_sampler=self.samplers[conc_type][split],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(data):
        """ Custom collate_fn to pass to dataloaders """        
        assert all(len(d)==2 for d in data)
        conc_type = data[0][1]

        assert all(isinstance(d[0], dict) for d in data)
        assert all(conc_type == d[1] for d in data)

        collated = {}
        for field in data[0][0]:
            assert all(field in d[0] for d in data)

            if field == "image":
                # PyTorch default collate function cannot recognize PIL Images
                collated[field] = [d[0][field] for d in data]
            elif field == "bboxes" or field == "bb_all":
                # Data may be of variable size
                collated[field] = [torch.tensor(d[0][field]) for d in data]
            elif field == "concept_label":
                # Let's leave em as-is...
                collated[field] = [d[0][field] for d in data]
            else:
                # Otherwise, process with default collate fn
                collated[field] = default_collate([d[0][field] for d in data])

        return collated, conc_type


class _SGGDataset(Dataset):
    def __init__(
        self, annotations, dataset_path, conc_type, conc_names, for_search_task=False
    ):
        super().__init__()
        self.annotations = annotations
        self.dataset_path = dataset_path
        self.conc_names = conc_names
        self.conc_type = conc_type
        self.for_search_task = for_search_task
    
    def __getitem__(self, idx):
        assert len(idx) == 3
        img_id, obj_ids, conc = idx

        # Return value
        data_dict = {}

        # Concept label; not directly consumed by model, for readability
        data_dict["concept_label"] = self.conc_names[conc]

        images_path = os.path.join(self.dataset_path, "images")
        vectors_path = os.path.join(self.dataset_path, "vectors")

        img_file = self.annotations[img_id]["file_name"]
        vecs_file = f"{img_file}.vectors"

        # Fetch and return raw image, bboxes and object indices
        if self.for_search_task or \
            not os.path.exists(os.path.join(vectors_path, vecs_file)):

            image_raw = os.path.join(images_path, img_file)
            image_raw = Image.open(image_raw)
            if image_raw.mode != "RGB":
                # Cast non-RGB images (e.g. grayscale) into RGB format
                old_image_raw = image_raw
                image_raw = Image.new("RGB", old_image_raw.size)
                image_raw.paste(old_image_raw)

            # Raw image
            data_dict["image"] = image_raw

            # Bounding boxes in the image
            oids = list(self.annotations[img_id]["annotations"])
            bboxes = [
                obj["bbox"] for obj in self.annotations[img_id]["annotations"].values()
            ]
            data_dict["bboxes"] = bboxes

            # Ind(s) of the designated object(s), with which to fetch bounding box(es)
            data_dict["bb_inds"] = tuple(oids.index(oid) for oid in obj_ids)

            if self.for_search_task:
                # Inds of *all* objects of the same concept in the same image; only
                # consumed in few-shot search task
                search_spec = list(zip(
                    (self.conc_type,) if type(self.conc_type) != tuple else self.conc_type,
                    (conc,) if type(conc) != tuple else conc
                ))
                data_dict["bb_all"] = [
                    oids.index(oid)
                    for oid, ann in self.annotations[img_id]["annotations"].items()
                    if all(c in ann[ctype] for ctype, c in search_spec)
                ]

        if os.path.exists(os.path.join(vectors_path, vecs_file)):
            # Fetch and return pre-computed feature vectors for the image
            vecs = torch.load(os.path.join(vectors_path, vecs_file))
            vecs = torch.stack([vecs[str(oid)] for oid in obj_ids], dim=0)
            data_dict["dec_out_cached"] = vecs
        
        return data_dict, self.conc_type
    
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
        with_replacement=False, for_search_task=False
    ):
        self.index_conc = index_conc
        self.hypernym_info = hypernym_info
        self.batch_size = batch_size
        self.num_exs_per_conc = num_exs_per_conc
        self.num_concepts = self.batch_size // self.num_exs_per_conc

        self.with_replacement = with_replacement
        self.for_search_task = for_search_task

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

                    if self.for_search_task:
                        # One few-shot episode per concept -- for time's interest...
                        concepts_to_del.add(conc)
                    else:
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


def _annotations_by_concept(annotations, metadata, conc_type, split):
    """
    Subroutine for filtering annotated images by specified concepts; returns
    filtered list of images & index by concepts
    """
    index_by_concept = {
        int(c): [
            (int(img), (obj_ids,) if type(obj_ids) else tuple(obj_ids))
            for img, insts_per_img in insts.items() for obj_ids in insts_per_img
        ]
        for c, insts in metadata[f"{conc_type}_instances"].items()
        if int(c) in metadata[f"{conc_type}_{split}_split"]
    }
    occurring_img_ids = set.union(*[
        {img_id for img_id, _ in insts} for insts in index_by_concept.values()
    ])
    annotations_filtered = {
        img["image_id"]: img for img in annotations
        if img["image_id"] in occurring_img_ids
    }

    # Object ids in annotation are read as str from metadata json; convert to int
    for img in annotations:
        img["annotations"] = {
            int(obj_id): ann for obj_id, ann in img["annotations"].items()
        }

    return index_by_concept, annotations_filtered
