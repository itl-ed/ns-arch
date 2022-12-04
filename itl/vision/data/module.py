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

        # Prepare dataloaders and samplers according to train/val/test split provided
        # in metadata
        self.datasets = {}; self.samplers = {}
        for conc_type in ["classes", "attributes"]: #, "relations"]:
            if len(metadata[f"{conc_type}_train_split"]) == 0 or \
                len(metadata[f"{conc_type}_val_split"]) == 0:
                continue

            self.datasets[conc_type] = {}
            self.samplers[conc_type] = {}

            hypernym_info = metadata[f"{conc_type}_hypernyms"] \
                if f"{conc_type}_hypernyms" in metadata else {}
            hypernym_info = { int(k): set(v) for k, v in hypernym_info.items() }

            if self.cfg.vision.task == "fs_classify":
                batch_size = self.cfg.vision.data.batch_size
                batch_size_eval = self.cfg.vision.data.batch_size_eval
            else:
                assert self.cfg.vision.task == "fs_search"
                batch_size = batch_size_eval = 1

            if stage in ["fit"]:
                # Training set required for "fit" stage setup
                index_train, ann_train = _annotations_by_concept(
                    annotations, metadata, conc_type, "train"
                )
                self.datasets[conc_type]["train"] = _SGGDataset(
                    ann_train, dataset_path, conc_type, metadata[f"{conc_type}_names"],
                    task_type=self.cfg.vision.task
                )
                self.samplers[conc_type]["train"] = _FewShotSGGDataSampler(
                    index_train, hypernym_info, batch_size,
                    self.cfg.vision.data.num_exs_per_conc,
                    task_type=self.cfg.vision.task
                )
            if stage in ["fit", "validate"]:
                # Validation set required for "fit"/"validate" stage setup
                index_val, ann_val = _annotations_by_concept(
                    annotations, metadata, conc_type, "val"
                )
                self.datasets[conc_type]["val"] = _SGGDataset(
                    ann_val, dataset_path, conc_type, metadata[f"{conc_type}_names"],
                    task_type=self.cfg.vision.task
                )
                self.samplers[conc_type]["val"] = _FewShotSGGDataSampler(
                    index_val, hypernym_info, batch_size_eval,
                    self.cfg.vision.data.num_exs_per_conc_eval,
                    task_type=self.cfg.vision.task,
                    with_replacement=False,
                    compress=self.cfg.vision.optim.compress_eval
                )
            if stage in ["test"]:
                # Test set required for "fit"/"test"/"predict" stage setup
                index_test, ann_test = _annotations_by_concept(
                    annotations, metadata, conc_type, "test"
                )
                self.datasets[conc_type]["test"] = _SGGDataset(
                    ann_test, dataset_path, conc_type, metadata[f"{conc_type}_names"],
                    task_type=self.cfg.vision.task
                )
                self.samplers[conc_type]["test"] = _FewShotSGGDataSampler(
                    index_test, hypernym_info, batch_size_eval,
                    self.cfg.vision.data.num_exs_per_conc_eval,
                    task_type=self.cfg.vision.task,
                    with_replacement=False,
                    compress=self.cfg.vision.optim.compress_eval
                )

        # Also setup class+attribute hybrid dataloader for few-shot search task
        if self.cfg.vision.task == "fs_search" and \
            "classes" in self.datasets and "attributes" in self.datasets:

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
                    task_type="fs_search"
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
                    cls_sampler.batch_size,
                    cls_sampler.num_exs_per_conc,
                    task_type="fs_search",
                    with_replacement=cls_sampler.with_replacement,
                    compress=cls_sampler.compress
                )

    def train_dataloader(self):
        return _ChainedLoader(*self._return_dataloaders("train"))
    
    def val_dataloader(self):
        return self._return_dataloaders("val")

    def test_dataloader(self):
        return self._return_dataloaders("test")

    def _return_dataloaders(self, split):
        return [
            self._fetch_dataloader(conc_type, split)
            for conc_type in self.datasets
        ]

    def _fetch_dataloader(self, conc_type, split):
        return DataLoader(
            dataset=self.datasets[conc_type][split],
            batch_sampler=self.samplers[conc_type][split],
            num_workers=self.cfg.vision.data.num_loader_workers,
            collate_fn=self._collate_fn,
            prefetch_factor=1
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
            elif field == "concept_label" or field == "bb_inds" or field == "supp_vecs":
                # Let's leave em as-is...
                collated[field] = [d[0][field] for d in data]
            else:
                # Otherwise, process with default collate fn
                collated[field] = default_collate([d[0][field] for d in data])

        return collated, conc_type


class _ChainedLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
    
    def __iter__(self):
        iters = [iter(dl) for dl in self.dataloaders]
        while True:
            for it in iters:
                yield next(it)


class _SGGDataset(Dataset):
    def __init__(
        self, annotations, dataset_path, conc_type, conc_names, task_type
    ):
        super().__init__()
        self.annotations = annotations
        self.dataset_path = dataset_path
        self.conc_names = conc_names
        self.conc_type = conc_type
        self.task_type = task_type
    
    def __getitem__(self, idx):
        images_path = os.path.join(self.dataset_path, "images")
        vectors_path = os.path.join(self.dataset_path, "vectors")

        if self.task_type == "fs_classify":
            assert len(idx) == 3
            img_id, obj_ids, conc = idx

            # Return value
            data_dict = {}

            # Concept label; not directly consumed by model, for readability
            data_dict["concept_label"] = self.conc_names[conc]

            img_file = self.annotations[img_id]["file_name"]
            vecs_file = f"{img_file}.vectors"

            # Fetch and return raw image, bboxes and object indices
            if not os.path.exists(os.path.join(vectors_path, vecs_file)):
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
                    ann["bbox"] for ann in self.annotations[img_id]["annotations"].values()
                ]
                data_dict["bboxes"] = bboxes

                # Ind(s) of the designated object(s), with which to fetch bounding box(es)
                data_dict["bb_inds"] = tuple(oids.index(oid) for oid in obj_ids)

            if os.path.exists(os.path.join(vectors_path, vecs_file)):
                # Fetch and return pre-computed feature vectors for the image
                vecs = torch.load(os.path.join(vectors_path, vecs_file))
                vecs = torch.stack([vecs[str(oid)] for oid in obj_ids], dim=0)
                data_dict["dec_out_cached"] = vecs
            
            return data_dict, self.conc_type
        
        if self.task_type == "fs_search":
            assert len(idx) == 2
            img_id, support_exs = idx

            # Return value
            data_dict = {}

            img_file = self.annotations[img_id]["file_name"]

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
                ann["bbox"] for ann in self.annotations[img_id]["annotations"].values()
            ]
            data_dict["bboxes"] = bboxes

            # Re-index concepts occurring in this item
            c2i = { c: i for i, c in enumerate(support_exs) }

            data_dict["bb_inds"] = [None] * len(c2i)
            data_dict["supp_vecs"] = [None] * len(c2i)
            for conc, exs in support_exs.items():
                if isinstance(self.conc_type, tuple):
                    conc_conds = list(zip(self.conc_type, conc))
                else:
                    conc_conds = [(self.conc_type, conc)]

                # Bounding boxes of all instances for each concept label, with respect
                # to the label space defined by the set of (re-indexed) concepts
                data_dict["bb_inds"][c2i[conc]] = [
                    oids.index(oid)
                    for oid, ann in self.annotations[img_id]["annotations"].items()
                    if all(ci in ann[ct] for ct, ci in conc_conds)
                ]

                # Support example vectors for each concept label
                ex_vecs = [
                    (self.annotations[ii]["file_name"], ois)
                    for ii, ois in exs
                ]
                ex_vecs = [
                    (torch.load(os.path.join(vectors_path, f"{fname}.vectors")), ois)
                    for fname, ois in ex_vecs
                ]
                ex_vecs = torch.stack([
                    torch.stack([all_vecs[str(oi)] for oi in ois], dim=0)
                    for all_vecs, ois in ex_vecs
                ], dim=0)

                data_dict["supp_vecs"][c2i[conc]] = ex_vecs

            return data_dict, self.conc_type
        
        raise ValueError("Invalid task type; shouldn't reach here")
    
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
        self, index_conc, hypernym_info, batch_size, num_exs_per_conc, task_type,
        with_replacement=True, compress=False
    ):
        self.index_conc = index_conc
        self.hypernym_info = hypernym_info
        self.batch_size = batch_size
        self.num_exs_per_conc = num_exs_per_conc
        self.num_concepts = self.batch_size // self.num_exs_per_conc
        self.task_type = task_type

        self.with_replacement = with_replacement
        self.compress = compress

        # Filter concepts to leave only those with more than K instances for use
        self.index_conc = {
            k: v for k, v in self.index_conc.items()
            if len(v) >= self.num_exs_per_conc
        }

        # Build ontology tree (forest) from contained concepts and provided
        # hypernym metadata info, find connected components; later we will
        # sample components first and then sample concepts from the components,
        # so as to avoid sampling from concepts belonging to connected ontology
        # tree
        self.ont_forest = nx.Graph()
        for c in self.index_conc:
            self.ont_forest.add_node(c)
            if c in self.hypernym_info:
                for h in self.hypernym_info[c]:
                    if h in self.index_conc:
                        self.ont_forest.add_edge(c, h)
        self.ont_forest = [
            list(comp) for comp in nx.connected_components(self.ont_forest)
        ]

        if self.task_type == "fs_classify":
            # Let's keep things simple by making batch size divisible by number of
            # exemplars per concept (equivalently, number of concepts or 'ways')
            assert self.batch_size % self.num_exs_per_conc == 0
        
        if self.task_type == "fs_search":
            # All images containing any instances of the concepts, tagged with
            # occurring concepts
            self.index_img = defaultdict(lambda: defaultdict(set))
            for conc, insts in self.index_conc.items():
                for img_id, obj_ids in insts:
                    self.index_img[img_id][conc].add(obj_ids)
            self.index_img = { k: dict(v) for k, v in self.index_img.items() }

    def __iter__(self):
        if self.task_type == "fs_classify":
            # Concept-based sampling of images for metric learning

            # For maintaining lists of concepts & exemplars as sampling candidates
            conc_ontology = copy.deepcopy(self.ont_forest)
            conc_exemplars = copy.deepcopy(self.index_conc)

            while True:
                if len(conc_ontology) < self.num_concepts:
                    # Exhausted, cannot sample anymore
                    return

                # First sample connected components in ontology, so as to avoid
                # sampling from more than one from same comp
                sampled_comps = random.sample(conc_ontology, self.num_concepts)

                # Then sample one concept from each sampled component
                sampled_concs = [random.sample(comp, 1)[0] for comp in sampled_comps]

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

        if self.task_type == "fs_search":
            # Concept-based sampling of images for few-shot search
            images = list(self.index_img)
            concepts_to_cover = set(self.index_conc)

            while True:
                if len(images) < self.batch_size:
                    return

                sampled_batch = []
                while len(sampled_batch) < self.batch_size and len(images) > 0:
                    if self.with_replacement:
                        # Make sure the each concept is included at most every C
                        # samples (where C is the total number of concepts)
                        if len(concepts_to_cover) == 0:
                            concepts_to_cover = set(self.index_conc)

                        smp_conc = random.sample(concepts_to_cover, 1)[0]
                        smp_img = random.sample(self.index_conc[smp_conc], 1)[0][0]
                    else:
                        smp_img = images.pop()

                        # If self.compress flag is set to True, skip images that
                        # only contain instances of concepts already covered, in
                        # the interest of time
                        if self.compress:
                            if len(set(self.index_img[smp_img]) & concepts_to_cover) == 0:
                                continue

                    # Make sure many-to-one mapping is avoided by allowing only one
                    # concept from each independent ontology tree
                    belonging_trees_inds = {
                        conc: [conc in tree for tree in self.ont_forest].index(True)
                        for conc in self.index_img[smp_img]
                        if any(conc in tree for tree in self.ont_forest)
                    }
                    concepts_by_belonging_trees = defaultdict(list)
                    for conc, ti in belonging_trees_inds.items():
                        concepts_by_belonging_trees[ti].append(conc)

                    sampled_concepts = [
                        random.sample(concs, 1)[0]
                        for concs in concepts_by_belonging_trees.values()
                    ]

                    # Sample (at most) K exemplars from other images for each concept
                    # included
                    support_exs = {}
                    for conc in sampled_concepts:
                        exs_from_other_imgs = [
                            (img_id, obj_ids)
                            for img_id, obj_ids in self.index_conc[conc]
                            if img_id != smp_img
                        ]
                        if len(exs_from_other_imgs) >= self.num_exs_per_conc:
                            exs_from_other_imgs = random.sample(
                                exs_from_other_imgs, self.num_exs_per_conc
                            )
                        else:
                            # Not enough support instances from other images!
                            concepts_to_cover -= {conc}
                            continue

                        support_exs[conc] = exs_from_other_imgs

                    if len(support_exs) == 0:
                        # Nothing to detect here
                        continue

                    concepts_to_cover -= set(sampled_concepts)
                    sampled_batch.append((smp_img, support_exs))

                if len(sampled_batch) < self.batch_size:
                    continue

                yield sampled_batch

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
