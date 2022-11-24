"""
Helper methods for preparing Visual Genome dataset for training/validation/testing,
to be called by prepare_data() in PyTorch Lightning DataModule.
"""
import os
import json
import logging
import zipfile
import requests
import multiprocessing
from itertools import product
from urllib.request import urlretrieve
from urllib.error import URLError
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import ijson
import torch
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError

logger = logging.getLogger(__name__)


VG_REPO_URL = "http://visualgenome.org/static/data/dataset"

def download_and_unzip(target_dir, target_filename):
    """
    Download a zip archive file containing a json file, extract, and remove the zip
    """
    url = f"{VG_REPO_URL}/{target_filename}"
    target_path = os.path.join(target_dir, target_filename)

    download_from_url(url, target_path)
    with zipfile.ZipFile(target_path) as zip_f:
        zip_f.extractall(target_dir)

    # Delete the downloaded zip archive
    os.remove(target_path)


def download_images(imgs_to_download):
    """
    Download image files listed in the argument list, each of whose entry is a pair
    of URL of the image and the path to store the downloaded image file
    """
    num_workers = multiprocessing.cpu_count()
    dl_queues = ThreadPool(num_workers).imap_unordered(
        _download_indiv_image, imgs_to_download
    )

    pbar = tqdm(enumerate(dl_queues), total=len(imgs_to_download))
    pbar.set_description("Downloading images")

    downloaded = 0
    for result in pbar:
        downloaded += result[1]
    logger.info(f"{downloaded} images downloaded")


def download_from_url(url, target_path):
    """
    Download a single file from the given url to the target path, with a progress bar shown
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))

    with open(target_path, 'wb') as file, tqdm(
        desc=target_path.split("/")[-1],
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def _download_indiv_image(url_and_path):
    """
    Helper method for downloading individual images, to be called from download_images()
    by multiple threads in parallel
    """
    url, path = url_and_path

    if not os.path.exists(path):
        try:
            urlretrieve(url, path)
        except URLError as e:
            logger.info(f"{e}: Retrying download {url}...")
            _download_indiv_image(url_and_path)
        except e:
            raise e

        return True
    else:
        return False


def process_image_metadata(target_dir, img_count):
    """
    Process the VG image metadata file ({target_dir}/image_data.json) to find image
    files to download, and then return a template dict to be filled with annotations.
    """
    # To be filled and stored
    img_anns = {}

    # Process metadata and download image files
    with open(f"{target_dir}/image_data.json") as imd_f:
        data = ijson.items(imd_f, "item")

        imgs_to_download = []

        for i, img in enumerate(data):

            if i>=img_count: break

            file_name = img["url"].split("/")[-1]
            file_path = os.path.join(target_dir, "images", file_name)

            imgs_to_download.append((img["url"], file_path))
            
            img_anns[img["image_id"]] = {
                "image_id": img["image_id"],
                "width": img["width"],
                "height": img["height"],
                "file_name": file_name
            }
    
    return img_anns, imgs_to_download


def reformat_annotations(target_dir, img_count, img_anns, seed):
    """
    Convert the original VG annotations into a simpler format, indexing concepts
    (class/attribute/relation) in order encountered. In effect, the re-formatting
    compresses the size of annotation from e.g. ~757MB to ~504MB.)
    """
    # Load attribute-to-synset map
    att2ss = {}
    with open(f"{target_dir}/attribute_synsets.json") as amap_f:
        data = ijson.kvitems(amap_f, "")

        for name, ss in data:
            att2ss[name] = ss

    # Mapping from predicate name to integer index
    cls_ids = defaultdict(lambda: len(cls_ids))
    att_ids = defaultdict(lambda: len(att_ids))
    rel_ids = defaultdict(lambda: len(rel_ids))

    # Indexing concept occurrences
    cls_occurrences = defaultdict(lambda: defaultdict(set))
    att_occurrences = defaultdict(lambda: defaultdict(set))
    rel_occurrences = defaultdict(lambda: defaultdict(set))

    # Indexing class/attribute predicate co-occurences; for use in few-shot search
    # training with cls-att composite search spec
    cls_att_cooccurrences = defaultdict(lambda: defaultdict(set))

    # Collect scene graph annotations and split
    with open(f"{target_dir}/scene_graphs.json") as sg_f, \
        open(f"{target_dir}/annotations.json", "w") as ann_f:

        data = ijson.items(sg_f, "item")

        pbar = tqdm(enumerate(data), total=img_count)
        pbar.set_description("Processing annotations")

        # Open json arrays output
        ann_f.write("[")

        for i, sg in pbar:

            if i>=img_count: break

            if i != 0:
                ann_f.write(", ")  # Comma delim except first entries

            img_id = sg["image_id"]

            # Reformat object annotations
            anno = {}
            for obj in sg["objects"]:

                obj_id = obj["object_id"]

                obj_ann = {
                    "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]]
                }

                classes = []
                for s in obj["synsets"]:
                    classes.append(cls_ids[s])
                    cls_occurrences[cls_ids[s]][img_id].add(obj_id)

                obj_ann["classes"] = classes

                if "attributes" in obj:
                    # Consider canonicalized attributes only
                    atts_valid = [a for a in obj["attributes"] if a in att2ss]

                    attributes = []
                    for a in atts_valid:
                        attributes.append(att_ids[att2ss[a]])
                        att_occurrences[att_ids[att2ss[a]]][img_id].add(obj_id)
                    
                    obj_ann["attributes"] = attributes
                else:
                    obj_ann["attributes"] = []

                anno[obj_id] = obj_ann

                # Bookkeeping class-attribute co-occurrences
                for ci, ai in product(obj_ann["classes"], obj_ann["attributes"]):
                    cls_att_cooccurrences[(ci, ai)][img_id].add(obj_id)

            # Insert relation annotations as well -- by relation subjects
            for r in sg["relationships"]:

                if "relations" not in anno[r["subject_id"]]:
                    anno[r["subject_id"]]["relations"] = set()

                if len(r["synsets"]) == 0:
                    if r["predicate"] == "OF":
                        # Relations with "OF" predicate do not have canonicalized synsets, presumably
                        # due to the fact that WordNet doesn't have "of" entry. We include a fake
                        # synset here that represents the "part-of" relationship, for it has high
                        # importance for our purpose
                        # (The *somewhat* 'inverse' relation of 'have' is represented by "have.v.01")
                        r["synsets"] = ["of.r.01"]
                    else:
                        # Otherwise skip relations with empty synsets
                        continue

                # Adding (relation predicate, object id) pairs as tuples to relation sets,
                # so as to remove duplicates (why do they have duplicates after all?)
                relations = []
                for s in r["synsets"]:
                    relations.append(rel_ids[s])
                    rel_occurrences[rel_ids[s]][img_id].add((obj_id, r["object_id"]))

                anno[r["subject_id"]]["relations"].add(
                    (tuple(relations), r["object_id"])
                )

            # Tuples back to lists
            for o in anno.values():
                if "relations" in o:
                    o["relations"] = [{
                        "relation": list(r[0]),
                        "object_id": r[1]
                    } for r in o["relations"] if len(r[0]) > 0]
                else:
                    o["relations"] = []

            entry = img_anns[img_id]
            entry.update({ "annotations": anno })

            ann_f.write(json.dumps(entry))
        
        # Close json arrays output
        ann_f.write("]")

    # Defaultdicts to dicts
    cls_ids = dict(cls_ids); att_ids = dict(att_ids); rel_ids = dict(rel_ids)
    cls_occurrences = {
        c: { img: list(insts) for img, insts in per_img.items() }
        for c, per_img in cls_occurrences.items()
    }
    att_occurrences = {
        a: { img: list(insts) for img, insts in per_img.items() }
        for a, per_img in att_occurrences.items()
    }
    rel_occurrences = {
        r: { img: list(insts) for img, insts in per_img.items() }
        for r, per_img in rel_occurrences.items()
    }
    cls_att_cooccurrences = {
        "_".join([str(c), str(a)]): {
            img: list(insts) for img, insts in per_img.items()
        }
        for (c, a), per_img in cls_att_cooccurrences.items()
    }

    # Check if each concept is hyper/hyponymy of another
    cls_hypernyms = {}; att_hypernyms = {}; rel_hypernyms = {}
    conc_ids_hypernyms = [
        (cls_ids, cls_hypernyms), (att_ids, att_hypernyms), (rel_ids, rel_hypernyms)
    ]
    for conc_ids, conc_hypernyms in conc_ids_hypernyms:
        for c in conc_ids:
            try:
                hypernyms_by_id = [
                    conc_ids[h._name] for h in wn.synset(c).hypernyms()
                    if h._name in conc_ids
                ]
                if len(hypernyms_by_id) > 0:
                    conc_hypernyms[conc_ids[c]] = hypernyms_by_id
            except WordNetError:
                # Likely only would apply to the custom relation concept "of.r.01"...
                logger.info(f"Synset '{c}' is not found in WordNet")

    # Store predicate indexing and hyper/hyponymy info and splits as metadata
    metadata = {
        "classes_names": _ind_dict_to_list(cls_ids),
        "attributes_names": _ind_dict_to_list(att_ids),
        "relations_names": _ind_dict_to_list(rel_ids),
        "classes_hypernyms": cls_hypernyms,
        "attributes_hypernyms": att_hypernyms,
        "relations_hypernyms": rel_hypernyms,
        "split_seed": seed
    }

    # Train/val/test split of dataset by *concepts* (not by images) for few-shot
    # training
    for conc_type in ["classes", "attributes", "relations"]:
        conc_names = metadata[f"{conc_type}_names"]

        split1 = int(len(conc_names) * 0.8)
        split2 = int(len(conc_names) * 0.9)

        concepts_shuffled = torch.randperm(len(conc_names)).tolist()

        metadata[f"{conc_type}_train_split"] = concepts_shuffled[:split1]
        metadata[f"{conc_type}_val_split"] = concepts_shuffled[split1:split2]
        metadata[f"{conc_type}_test_split"] = concepts_shuffled[split2:]

    # All witnessed concept (pair) instances
    metadata["classes_instances"] = cls_occurrences
    metadata["attributes_instances"] = att_occurrences
    metadata["relations_instances"] = rel_occurrences
    metadata["classes_attributes_pair_instances"] = cls_att_cooccurrences

    # Write metadata to file
    with open(f"{target_dir}/metadata.json", "w") as meta_f:
        json.dump(metadata, meta_f)


def _ind_dict_to_list(d):

    ret = [None] * len(d)

    for v, i in d.items():
        assert type(i) == int
        ret[i] = v

    return ret
