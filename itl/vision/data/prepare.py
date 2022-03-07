"""
Methods for preparing Visual Genome dataset for training/validation/testing, to be
called by prepare_data() in PyTorch Lightning DataModule. Includes methods for:

1) Download and extract necessary files (metadata & scene graph annotations) from
    the Visual Genome repository (http://visualgenome.org/static/data/dataset/...)
2) Assemble & reformat the annotations into the format expected by our custom
    detectron2 data reader, making sure predicates are properly canonicalized
    (esp. attributes). Make train/val/test splits along the way.
3) Download actual image files from the URLs included in the fetched metadata
"""
import os
import json
import zipfile
import urllib.request
import multiprocessing
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import ijson
from tqdm import tqdm

from ..utils import download_from_url


VG_TOTAL_COUNT = 108077  # Visual genome data stat

def download_and_unzip(url, target_dir, target_filename):
    """
    Download a zip archive file containing a json file, extract, and remove the zip
    """
    target_path = os.path.join(target_dir, target_filename)

    download_from_url(url, target_path)
    with zipfile.ZipFile(target_path) as zip_f:
        zip_f.extractall(target_dir)

    # Delete the downloaded zip archive
    os.remove(target_path)


def process_image_metadata(target_dir, img_count):
    """
    Process the VG image metadata file ({target_dir}/image_data.json) to download
    image files, and then return a template dict to be filled with annotations.
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
                "file_name": file_path
            }
    
    return img_anns, imgs_to_download


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


def _download_indiv_image(url_and_path):
    """
    Helper method for downloading individual images, to be called from download_images()
    by multiple threads in parallel
    """
    url, path = url_and_path

    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        return True
    else:
        return False


def reformat_annotations(target_dir, img_count, img_anns):
    """
    Convert the original VG annotations into the format expected by our detectron2
    data reader, while making train/val/test split.
    
    (In effect, the re-formatting compresses the size of training/testing data
    from ~757MB to ~504MB.)
    """
    train_cnt = val_cnt = test_cnt = 0

    obj_counts = 0
    obj_pair_counts = 0       # Ordered pairs

    cls_counts = defaultdict(int)
    att_counts = defaultdict(int)
    rel_counts = defaultdict(int)

    # Load attribute-to-synset map
    attr2ss = {}
    with open(f"{target_dir}/attribute_synsets.json") as amap_f:
        data = ijson.kvitems(amap_f, "")

        for name, ss in data:
            attr2ss[name] = ss
    
    # First pass to collect counts of predicate occurences
    with open(f"{target_dir}/scene_graphs.json") as sg_f:
        data = ijson.items(sg_f, "item")

        pbar = tqdm(enumerate(data), total=VG_TOTAL_COUNT)
        pbar.set_description("Collecting statistics")

        for i, sg in pbar:
            obj_counts += len(sg["objects"])
            obj_pair_counts += len(sg["objects"]) * (len(sg["objects"]) - 1)

            for obj in sg["objects"]:
                for s in obj["synsets"]:
                    cls_counts[s] += 1
                if "attributes" in obj:
                    # Consider canonicalized attributes only
                    attrs = [a for a in obj["attributes"] if a in attr2ss]
                    for a in attrs:
                        att_counts[attr2ss[a]] += 1

            for r in sg["relationships"]:
                # Read the comment below re. treatment of "OF" relation predicate
                if len(r["synsets"]) == 0:
                    if r["predicate"] == "OF":
                        r["synsets"] = ["of.r.01"]

                for s in r["synsets"]:
                    rel_counts[s] += 1
    
    # Filter predicates to leave only those with at least 100 occurrences
    # (in the whole VGdataset)
    cls_hifreq = {k for k, v in cls_counts.items() if v >= 100}
    att_hifreq = {k for k, v in att_counts.items() if v >= 100}
    rel_hifreq = {k for k, v in rel_counts.items() if v >= 100}

    # Mapping from predicate name to integer index
    cls_ids = defaultdict(lambda: len(cls_ids))
    attr_ids = defaultdict(lambda: len(attr_ids))
    rel_ids = defaultdict(lambda: len(rel_ids))

    # Collect scene graph annotations and split
    with open(f"{target_dir}/scene_graphs.json") as sg_f, \
        open(f"{target_dir}/annotations_train.json", "w") as train_f, \
        open(f"{target_dir}/annotations_val.json", "w") as val_f, \
        open(f"{target_dir}/annotations_test.json", "w") as test_f:

        data = ijson.items(sg_f, "item")

        pbar = tqdm(enumerate(data), total=img_count)
        pbar.set_description("Processing annotations")

        # Open json arrays output
        train_f.write("[")
        val_f.write("[")
        test_f.write("[")

        for i, sg in pbar:

            if i>=img_count: break

            # 8:1:1 train/val/test split
            if i % 10 < 8:
                out_f = train_f
                train_cnt += 1
            elif i % 10 == 8:
                out_f = val_f
                val_cnt += 1
            else:
                out_f = test_f
                test_cnt += 1
            
            if not (i == 0 or i == 8 or i == 9):
                out_f.write(", ")  # Comma delim except first entries

            img_id = sg["image_id"]

            # Reformat object annotations
            anno = {}
            for obj in sg["objects"]:

                obj_id = obj["object_id"]

                obj_ann = {
                    "object_id": obj_id,
                    "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                    "classes": [cls_ids[s] for s in obj["synsets"] if s in cls_hifreq]
                }

                if "attributes" in obj:
                    # Consider canonicalized attributes only
                    attrs = [a for a in obj["attributes"] if a in attr2ss]
                    obj_ann["attributes"] = [attr_ids[attr2ss[a]]
                        for a in attrs if attr2ss[a] in att_hifreq]
                else:
                    obj_ann["attributes"] = []

                anno[obj_id] = obj_ann

            # Insert relation annotations as well -- by relation subjects
            for r in sg["relationships"]:

                if "relations" not in anno[r["subject_id"]]:
                    anno[r["subject_id"]]["relations"] = set()

                if len(r["synsets"]) == 0:
                    if r["predicate"] == "OF":
                        # Relations with "OF" predicate do not have canonicalized synsets, presumably
                        # due to the fact that WordNet doesn't have "of" entry (because it's open-class)
                        #
                        # However, we would like to include a fake synset here that represents the 
                        # "part-of" relationship, for it has high importance for our purpose
                        # (The *somewhat* 'inverse' relation of 'has' is represented by "have.v.01")

                        r["synsets"] = ["of.r.01"]
                    else:
                        # Otherwise skip relations with empty synsets
                        continue

                # Adding (relation predicate, object id) pairs as tuples to relation sets,
                # so as to remove duplicates (why do they have duplicates after all?)
                anno[r["subject_id"]]["relations"].add((
                    tuple([rel_ids[s] for s in r["synsets"] if s in rel_hifreq]),
                    r["object_id"]
                ))

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
            entry.update({ "annotations": list(anno.values()) })

            out_f.write(json.dumps(entry))
        
        # Close json arrays output
        train_f.write("]")
        val_f.write("]")
        test_f.write("]")

    # Store predicate indexing as metadata
    with open(f"{target_dir}/metadata.json", "w") as meta_f:
        classes = _ind_dict_to_list(cls_ids)
        attributes = _ind_dict_to_list(attr_ids)
        relations = _ind_dict_to_list(rel_ids)

        metadata = {
            "train_cnt": train_cnt,
            "val_cnt": val_cnt,
            "test_cnt": test_cnt,
            "train_ann_path": f"annotations_train.json",
            "val_ann_path": f"annotations_val.json",
            "test_ann_path": f"annotations_test.json",
            "classes": classes,  # **
            "attributes": attributes,
            "relations": relations,
            "obj_counts": obj_counts,
            "obj_pair_counts": obj_pair_counts,
            "classes_counts": [cls_counts[c] for c in classes],
            "attributes_counts": [att_counts[a] for a in attributes],
            "relations_counts": [rel_counts[r] for r in relations]
        }
        # ** Using 'classes' instead of 'thing_classes' in order to circumvent
        # 'print_instances_class_histogram' method (called inside 'build_~_loader' methods)

        json.dump(metadata, meta_f)


def _ind_dict_to_list(d):

    ret = [None] * len(d)

    for v, i in d.items():
        assert type(i) == int
        ret[i] = v

    return ret
