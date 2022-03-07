"""
For sampling few-shot detection episodes: an IterableDataset which samples N-way K-shot
few-shot detection episodes from pre-registered dataset, for object classes, attributes
and relations -- one in each sampled entry
"""
import os
import random
from collections import defaultdict

import ijson
from tqdm import tqdm
from torch.utils.data import IterableDataset
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode


class FewShotDataset(IterableDataset):
    """
    In this dataset, each 'instance' is actually a few-shot detection episode. At each
    next() call, sample N predicates (for each type of category -- class, attribute,
    relation) and then I (|S|+|Q|) examples (image+bbox) per predicate.

    (K is not necessary in neighborhood component analysis (NCA) loss computation, making
    it an unused hyperparameter)
    """
    def __init__(self, dataset_name, ep_params, is_train=True, eval_seed=0):
        # N: Number of 'ways' in each batch
        # K: Number of shots; i.e. size of support set, or |S|
        # I: Total number of instances in each batch; i.e. |S|+|Q|
        self.N, self.K, self.I = ep_params
        if not (self.I - self.K > 0): raise ValueError
        
        self.is_train = is_train
        self.eval_seed = eval_seed       # For controlled evaluation

        # Index images by occurring instances of each concept, so that they can
        # be sampled for few-shot training
        md = MetadataCatalog.get(dataset_name)

        self.img_info = {}

        instances_cls = defaultdict(list)
        instances_att = defaultdict(list)
        instances_rel = defaultdict(list)

        with open(os.path.join(md.data_dir, md.ann_path)) as anno_f:
            data = ijson.items(anno_f, "item")

            pbar = tqdm(data, total=md.count)
            pbar.set_description(f"Indexing imgs by predicates ({dataset_name})")

            for img in pbar:
                img_id = img["image_id"]

                self.img_info[img_id] = {
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"]
                }
                oid_to_id = {
                    obj["object_id"]: oi for oi, obj in enumerate(img["annotations"])
                }

                for obj in img["annotations"]:
                    for c in obj["classes"]:
                        instances_cls[c].append(
                            (img_id, obj["object_id"], obj["bbox"])
                        )
                    for a in obj["attributes"]:
                        instances_att[a].append(
                            (img_id, obj["object_id"], obj["bbox"])
                        )
                    for rs in obj["relations"]:
                        for r in rs["relation"]:
                            # Fetch relation arg2 obj
                            arg2 = oid_to_id[rs["object_id"]]
                            arg2 = img["annotations"][arg2]

                            instances_rel[r].append(
                                (img_id,
                                    (obj["object_id"], rs["object_id"]),
                                    (obj["bbox"], arg2["bbox"]))
                            )
        
        self.instances = {
            "cls": instances_cls,
            "att": instances_att,
            "rel": instances_rel
        }
        self.sampleables = {
            cat_type: [k for k, v in insts.items() if len(v)>self.I]
            for cat_type, insts in self.instances.items()
        }

    def __iter__(self):
        """
        For training, yield from infinite stream of randomly sampled N-way K-shot episodes.
        
        For evaluation, iterate over all categories in order to produce episodes consisting
        of N categories (e.g. [0,1,2,3,4], [5,6,7,8,9], ... for N=5), sampling K support
        instances and I-K query instances for each category. (Omit last episodes with less
        than N categories.)
        """
        if self.is_train:
            while True:
                episodes = {
                    cat_type: self._sample_insts(
                        random.sample(self.sampleables[cat_type], self.N),
                        cat_type
                    )
                    for cat_type in ["cls", "att", "rel"]
                }
                episodes["shot"] = self.K
                yield episodes
        else:
            random.seed(self.eval_seed)

            # Random permutations of categories
            shuffled_sampleables = {
                cat_type: random.sample(cats, len(cats))
                for cat_type, cats in self.sampleables.items()
            }

            for cat_type in ["cls", "att", "rel"]:
                # Yield N-way K-shot episodes of single category type (one of cls/att/rel) with
                # K support instances & I-K query instances for each category
                while len(shuffled_sampleables[cat_type]) >= self.N:
                    sampled_cats = shuffled_sampleables[cat_type][:self.N]
                    shuffled_sampleables[cat_type] = shuffled_sampleables[cat_type][self.N:]

                    yield {
                        cat_type: self._sample_insts(sampled_cats, cat_type),
                        "shot": self.K
                    }
    
    def __len__(self):
        """Dataset length defined only in context of evaluation"""
        if self.is_train:
            return
        else:
            return sum([len(cats) // self.N for cats in self.sampleables.values()])

    def _sample_insts(self, sampled_cats, cat_type):
        """
        Helper method that sample I instances (i.e. image with corresponding bbox(es))
        for each sampled category from given lists, making sure sampled instances
        are non-instances of other categories
        """
        episode = defaultdict(list)

        insts_sampled = set()    # Identifiers of all objects sampled so far
        for c in sampled_cats:
            while len(episode[c]) < self.I:
                insts = random.sample(self.instances[cat_type][c], self.I-len(episode[c]))
                insts_new = [
                    {
                        **self.img_info[i[0]],
                        "annotations": _reformat((i[1], i[2]), c, cat_type)
                    }
                    for i in insts if (i[0], i[1]) not in insts_sampled
                ]
                insts_sampled |= {(i[0], i[1]) for i in insts}

                episode[c] += insts_new

        return dict(episode)


def _reformat(info, c, cat_type):
    """Reshape annotation info format accepted by mapper"""
    oi, bbox = info

    if cat_type == "cls":
        return [{
            "object_id": oi,
            "bbox": bbox,
            "bbox_mode": BoxMode.XYWH_ABS,
            "classes": [c],
            "atributes": [],
            "relations": []
        }]
    elif cat_type == "att":
        return [{
            "object_id": oi,
            "bbox": bbox,
            "bbox_mode": BoxMode.XYWH_ABS,
            "classes": [],
            "atributes": [c],
            "relations": []
        }]
    elif cat_type == "rel":
        return [{
            "object_id": oi[0],
            "bbox": bbox[0],
            "bbox_mode": BoxMode.XYWH_ABS,
            "classes": [],
            "atributes": [],
            "relations": [{ "relation": [c], "object_id": oi[1] }]
        }, {
            "object_id": oi[1],
            "bbox": bbox[1],
            "bbox_mode": BoxMode.XYWH_ABS,
            "classes": [],
            "atributes": [],
            "relations": []
        }]
    else:
        raise ValueError("Why are you here")
