"""
Our custom evaluator classes for evaluating scene graph generation models with
Visual Genome dataset. Implements two modes of evaluation:

    1) Object detection mode, where the model is given images only and needs to
        detect (propose) object bboxes prior to recognizing appropriate predicates.
    2) Predicate classification mode, where the model does not have proposal generator
        module and is given some set of proposal boxes along with images.
"""
import os
import json
import uuid
from itertools import chain

import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.structures import Boxes, BoxMode
from detectron2.evaluation import DatasetEvaluator


class _VGBatchEvaluator(DatasetEvaluator):
    """
    Abstract class containing common codes for VGDetEvaluator and VGClfEvaluator
    """
    def __init__(self, dataset_name, temp_dir):
        self._dataset_name = dataset_name
        md = MetadataCatalog.get(self._dataset_name)

        self._predicates = {
            "cls": md.classes,
            "att": md.attributes,
            "rel": md.relations
        }
        self._ann_path = os.path.join(md.data_dir, md.ann_path)
        self._cpu_device = torch.device("cpu")

        self._temp_dir = os.path.join(temp_dir, "temp")
        os.makedirs(self._temp_dir, exist_ok=True)

    def reset(self):
        """
        Collect bbox info in memory, prediction scores in temp files (disk).
        """
        self._predictions = {
            "bbox": [],
            # Temporary file paths
            # (Could use tempfile package and pass around file handles directly, but such
            # handles (_io.BufferedReader type) cannot be pickled)
            "cls": os.path.join(self._temp_dir, "vgev_cls_" + str(uuid.uuid4().hex)),
            "att": os.path.join(self._temp_dir, "vgev_att_" + str(uuid.uuid4().hex)),
            "rel": os.path.join(self._temp_dir, "vgev_rel_" + str(uuid.uuid4().hex))
                if self._rel_predict else None
        }

    def process(self, inputs, outputs):
        """
        Aggregating predictions, where we can choose to include or ignore relation
        predictions in addition to class & attribute predictions.
        """
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor
            cls_scores = instances.pred_objectness * instances.pred_classes
            att_scores = instances.pred_objectness * instances.pred_attributes

            if self._rel_predict:
                rel_scores = (instances.pred_objectness * instances.pred_objectness.T)
                rel_scores = rel_scores[:,:,None] * instances.pred_relations
            else:
                rel_scores = None

            self._predictions["bbox"].append((image_id, boxes))

            self._store_scores(cls_scores, self._predictions["cls"])
            self._store_scores(att_scores, self._predictions["att"])
            if self._rel_predict:
                self._store_scores(rel_scores, self._predictions["rel"])

    def evaluate(self):
        """
        Collect prediction info from all ranks, compute evaluation metrics as required by
        evaluation mode (det or clf), close tempfiles
        """
        comm.synchronize()
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return {}, {}

        # Aggregate data in each field into a single iterator
        tmp_files = {
            "cls": [open(prd["cls"], "rb") for prd in all_predictions],
            "att": [open(prd["att"], "rb") for prd in all_predictions],
            "rel": [open(prd["rel"], "rb") for prd in all_predictions]
                if self._rel_predict else None
        }
        predictions = {
            "bbox": sum([prd["bbox"] for prd in all_predictions], []),
            "cls": chain.from_iterable(tmp_files["cls"]),
            "att": chain.from_iterable(tmp_files["att"]),
            "rel": chain.from_iterable(tmp_files["rel"])
                if self._rel_predict else None
        }
        del all_predictions

        metrics = self._compute_metrics(predictions)

        cat_types = ["cls", "att", "rel"] if self._rel_predict else ["cls", "att"]
        for cat_type in cat_types:
            for tf in tmp_files[cat_type]:
                tf.close(); os.remove(tf.name)
        
        return metrics
    
    def _compute_metrics(self):
        raise NotImplementedError

    def _store_scores(self, scores, b_file_path):
        """
        Flatten given tensors containing scores into a 1-D list, and write to given binary file
        """
        scores = scores.view(-1).tolist()
        scores = map(_approx_score, scores)
        scores = (",".join(scores) + "\n").encode()

        with open(b_file_path, "ab") as b_f:
            b_f.write(scores)


class VGDetEvaluator(_VGBatchEvaluator):
    """
    Evaluate object detection performance with the mean average precision (mAP) metric.
    Assumes the visual model is only provided with raw images, and output instances are
    first generated by proposal generators and then tuned by RoI heads.
    """
    def __init__(self, dataset_name, temp_dir):
        super().__init__(dataset_name, temp_dir)
        self._rel_predict = False
    
    def _compute_metrics(self, predictions):
        """Compute mean average precision (mAP)"""
        # Shorthands; N = number of predictions made, C = number of categories
        N = [len(bs) for _, bs in predictions["bbox"]]
        C = {cat_type: len(cats) for cat_type, cats in self._predicates.items()}

        # Read and process ground-truth annotations
        with open(self._ann_path) as ann_f:
            gt_ann = json.load(ann_f)
            gt_ann = {img_ann["image_id"]: img_ann["annotations"]
                for img_ann in gt_ann}

        for img_id in gt_ann:
            ann = gt_ann[img_id]
            ann_processed = {
                "len": len(ann),
                "bbox": Boxes(BoxMode.convert(
                    np.stack([a["bbox"] for a in ann]), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
                )) if len(ann) > 0 else Boxes([]),
                "cls": torch.stack(
                    [_1hot(a["classes"], C["cls"]).sum(dim=0) for a in ann]
                ) if len(ann) > 0 else torch.empty((0, C["cls"])),
                "att": torch.stack(
                    [_1hot(a["attributes"], C["att"]).sum(dim=0) for a in ann]
                ) if len(ann) > 0 else torch.empty((0, C["att"]))
            }
            gt_ann[img_id] = ann_processed
        
        # Total counts of occurences of each predicates in whole tested data
        gt_cls_cnts = [gt_ann[img]["cls"] for img, _ in predictions["bbox"]]
        gt_cls_cnts = torch.cat(gt_cls_cnts).sum(dim=0)
        gt_att_cnts = [gt_ann[img]["att"] for img, _ in predictions["bbox"]]
        gt_att_cnts = torch.cat(gt_att_cnts).sum(dim=0)
        
        # Compute pairwise IoUs btw. ground truth vs. predictions for each image,
        # and match each prediction with ground truth object by max IoU
        iou_maxs = [(img, box_iou(bxs, gt_ann[img]["bbox"].tensor))
            for img, bxs in predictions["bbox"]]
        iou_maxs = [
            (img, torch.max(ious, dim=1))
                if gt_ann[img]["len"] > 0 else (img, _NullMax(len(ious)))
            for img, ious in iou_maxs
        ]    # img_id info still needed in order to fetch corresponding ground truth annos

        # Filtering predictions with IoU smaller than threshold
        iou_thresholds = [0.5, 0.75]
        iou_filters = [
            (thresh, torch.cat([max_gt.values > thresh for _, max_gt in iou_maxs]))
            for thresh in iou_thresholds
        ]

        # Mapping from prediction to matched (i.e. max iou) ground-truth object; this info
        # will be used later to locate any duplicate detections on same gt objects
        map_gt_idxs = [(gt_ann[img]["len"], max_gt.indices) for img, max_gt in iou_maxs]

        # Annotations of mapped ground-truth objects
        mapped_gt_cls = torch.cat([
            torch.gather(gt_ann[img]["cls"], 0, max_gt.indices[:,None].expand(-1, C["cls"]))
                if gt_ann[img]["len"] > 0
                else torch.zeros((len(max_gt.values), C["cls"]), dtype=gt_ann[img]["cls"].dtype)
            for img, max_gt in iou_maxs
        ])
        mapped_gt_att = torch.cat([
            torch.gather(gt_ann[img]["att"], 0, max_gt.indices[:,None].expand(-1, C["att"]))
                if gt_ann[img]["len"] > 0
                else torch.zeros((len(max_gt.values), C["att"]), dtype=gt_ann[img]["att"].dtype)
            for img, max_gt in iou_maxs
        ])

        cls_APs, cls_mAPs = self._compute_mAP(
            "cls", N, C["cls"], predictions["cls"], gt_cls_cnts,
            iou_filters, mapped_gt_cls, map_gt_idxs
        )
        att_APs, att_mAPs = self._compute_mAP(
            "att", N, C["att"], predictions["att"], gt_att_cnts,
            iou_filters, mapped_gt_att, map_gt_idxs
        )

        return {"bbox": {**cls_APs, **att_APs}}, {"bbox": {**cls_mAPs, **att_mAPs}}
    
    @staticmethod
    def _compute_mAP(
        cat_type, N, C, pred_files, gt_category_cnts,
        iou_filters, mapped_gt_category, map_gt_idxs
    ):
        """
        Read scores from prediction temp files, sort category-wise by scores, prepare
        precision-recall curves, aggregate into summary stat
        """
        metrics = {
            thresh: [None] * C for thresh, _ in iou_filters
        }   # Value to return

        # Read prediction scores from stored temporary disk file
        scores_per_img = [
            np.array(scores.split(b","), dtype=np.int16).reshape(N[i], -1)
                if len(scores.strip()) > 0
                else np.empty((0, C), dtype=np.int16)
            for i, scores in enumerate(pred_files)
        ]
        scores_all = np.concatenate(scores_per_img)

        pbar = tqdm(range(C), total=C)
        pbar.set_description(f"Computing APs ({cat_type})")

        # Calculate average precision for each possible category
        # (I could do this not sequentially but rather in parallel by manipulating
        # 2-D tensors, but there are WAY too many predicates (even after pruning
        # low-freq ones) and it hogs up the memory to do that)
        for c in pbar:
            sort_inds_per_img = [np.argsort(-ss[:,c]) for ss in scores_per_img]
            inv_sort_inds_per_img = [
                np.argsort(ids) for ids in sort_inds_per_img
            ]    # For mapping duplicate filters back to original order
            sort_inds_all = np.argsort(-scores_all[:,c])

            # Per PASCAL-VOC style eval, suppress any predictions that are mapped to same
            # ground-truth objects and do not have non-best scores. These duplicate filters
            # are defined per category.
            map_gt_idxs_sorted = [
                (N_gt, mapping[ids] if N_gt > 0 else torch.empty((0), dtype=torch.long))
                for ids, (N_gt, mapping) in zip(sort_inds_per_img, map_gt_idxs)
            ]
            map_gt_idxs_sorted = [
                _1hot(mapping, N_gt) for (N_gt, mapping) in map_gt_idxs_sorted
            ]
            dupl_filters_sorted = [
                1 - (mapping_st * (mapping_st.cumsum(dim=0)>1)).sum(dim=1)
                for mapping_st in map_gt_idxs_sorted
            ]
            dupl_filters = [
                flt[ids] if len(flt) > 0 else torch.zeros(len(ids), dtype=flt.dtype)
                for flt, ids in zip(dupl_filters_sorted, inv_sort_inds_per_img)
            ]
            dupl_filters = torch.cat(dupl_filters)

            ## Finally compute mAP, for each IoU threshold

            # Test if each prediction is TP or FP by combining all three filters;
            # a prediction is TP only if category is correct, IoU is over threshold,
            # and is not non-best duplicate
            pred_is_TP = torch.stack([flt for _, flt in iou_filters], dim=1) * \
                mapped_gt_category[:,c][:,None] * dupl_filters[:,None]
            pred_is_TP = pred_is_TP[sort_inds_all]

            # Compute precision-recall curve(s)
            TPs = pred_is_TP.cumsum(dim=0)
            FPs = (1 - pred_is_TP).cumsum(dim=0)
            precision = TPs / (TPs+FPs)
            recall = TPs / gt_category_cnts[c]

            # Fill in with computed mAPs
            for i, thresh in enumerate(iou_filters):
                metrics[thresh[0]][c] = _AP_from_PR(
                    precision[:,i], recall[:,i]
                )

        # Return collections of AP values, along with their averages (i.e. mAP); for the
        # latter, promptly disregard None values (i.e. categories with no ground-truth
        # occurrences)
        metrics_per_cat = {
            f"{cat_type}_AP@{thresh}": APs
            for thresh, APs in metrics.items()
        }
        metrics_avg = {
            field.replace("AP", "mAP"): np.mean([ap for ap in APs if ap is not None])
            for field, APs in metrics_per_cat.items()
        }

        return metrics_per_cat, metrics_avg


class VGClfEvaluator(_VGBatchEvaluator):
    """
    Evaluate predicate classification performance with the recall@K metric. Assumes the
    visual model is provided with some set of ground-truth region proposals along with
    raw images, and needs to predict object classes, attributes and relations between
    object pairs.
    """
    def __init__(self, dataset_name, temp_dir):
        super().__init__(dataset_name, temp_dir)
        self._rel_predict = True

    def _compute_metrics(self, predictions):
        """
        Compute recall@K; i.e. the fraction of ground-truth occurrences which are
        successfully included in top K predictions. Can be aggregated over all categories,
        or aggregated per category first and then take average -- report both.
        """
        # Shorthands; N = number of predictions made, C = number of categories
        # (Here N is also equal to the number of ground truth objects)
        N = [len(bs) for _, bs in predictions["bbox"]]
        C = {cat_type: len(cats) for cat_type, cats in self._predicates.items()}

        # Read and process ground-truth annotations
        with open(self._ann_path) as ann_f:
            gt_ann = json.load(ann_f)
            gt_ann = {img_ann["image_id"]: img_ann["annotations"]
                for img_ann in gt_ann}

        # Sequentially process predictions on per-image basis (as opposed to mAP computation
        # in VGDetEvaluator), as one-hot relation matrices are large (N*N*C_R) and it will
        # quickly exhaust memory spaces to manipulate them all at once
        preds_per_img = zip(
            predictions["bbox"], predictions["cls"], predictions["att"], predictions["rel"]
        )
        pbar = tqdm(enumerate(preds_per_img), total=len(N))
        pbar.set_description("Computing recall@Ks")

        ks = [50, 100]
        stats = {
            cat_type: {
                "per_cat": torch.zeros((C[cat_type], len(ks))),
                "per_img": torch.zeros(len(ks)),
                "cat_exists": torch.zeros(C[cat_type]),
                "img_with_pos": 0
            }
            for cat_type in ["cls", "att", "rel"]
        }    # Recall values to aggregate

        for i, ((img_id, _), cls_scores, att_scores, rel_scores) in pbar:
            ann = gt_ann[img_id]

            if len(ann) == 0:
                # No stats to collect
                continue

            oid_map = {a["object_id"]: i for i, a in enumerate(ann)}
            ann_processed = {
                "len": len(ann),
                "bbox": Boxes(BoxMode.convert(
                    np.stack([a["bbox"] for a in ann]), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
                )) if len(ann) > 0 else Boxes([]),
                "cls": torch.stack(
                    [_1hot(a["classes"], C["cls"]).sum(dim=0) for a in ann]
                ) if len(ann) > 0 else torch.empty((0, C["cls"])),
                "att": torch.stack(
                    [_1hot(a["attributes"], C["att"]).sum(dim=0) for a in ann]
                ) if len(ann) > 0 else torch.empty((0, C["att"])),
                "rel": torch.stack([
                    sum([
                        _1hot(oid_map[r["object_id"]], len(ann))[:,None] * \
                        _1hot(r["relation"], C["rel"]).sum(dim=0)[None,:]
                        for r in a["relations"]
                    ], torch.zeros(len(ann), C["rel"]))
                    for a in ann
                ]).reshape(-1, C["rel"]) if len(ann) > 0 else torch.empty((0, C["rel"]))
            }

            # Ensuring the orders of bboxes agree between annotations vs. predictions
            # by comparing bbox coordinates; rounding noises may take place so tolerate
            # 'minimal' differences (i.e. at most BOX_EPS)
            # BOX_EPS = 1
            # bbox_coords_diff = ann_processed["bbox"].tensor - bbox.round()
            # assert (bbox_coords_diff > BOX_EPS).sum() == 0

            num_pred = N[i]
            if num_pred > 0:
                scores = {
                    "cls": np.array(cls_scores.strip().split(b","), dtype=np.int16),
                    "att": np.array(att_scores.strip().split(b","), dtype=np.int16),
                    "rel": np.array(rel_scores.strip().split(b","), dtype=np.int16)
                }
            else:
                scores = {
                    "cls": np.empty(0, dtype=np.int16),
                    "att": np.empty(0, dtype=np.int16),
                    "rel": np.empty(0, dtype=np.int16)
                }

            for cat_type in ["cls", "att", "rel"]:
                # First check if there's any positive annotations
                gt_total_per_cat = ann_processed[cat_type].sum(dim=0)
                gt_total_per_img = gt_total_per_cat.sum(dim=0)

                if gt_total_per_img == 0:
                    # No stats to collect
                    continue

                # Sort mappings by score & inverse mappings back to original order
                sort_inds = np.argsort(-scores[cat_type])
                inv_sort_inds = np.argsort(sort_inds)

                # Filter top K predictions
                topk_filters = np.stack([
                    np.pad(np.ones(k, dtype=np.int16), (0, len(scores[cat_type])-k))
                    for k in ks
                ], axis=1)
                topk_filters = topk_filters[inv_sort_inds]
                topk_filters = torch.tensor(topk_filters.reshape(-1, C[cat_type], 2))

                # Ground truths that successfully made into top k lists
                recovered = topk_filters * ann_processed[cat_type][:,:,None]

                # Compute recall@K per overall image, and per category; recall per image
                # is indifferent to category of each ground-truth, while recall per category
                # is more sensitive to rare categories with lower prediction accuracy
                recovered_per_cat = recovered.sum(dim=0)
                recovered_per_img = recovered_per_cat.sum(dim=0)

                recall_per_cat = recovered_per_cat / gt_total_per_cat[:,None]
                recall_per_img = recovered_per_img / gt_total_per_img

                # Aggregate to total
                stats[cat_type]["per_cat"] += torch.nan_to_num(recall_per_cat)
                stats[cat_type]["per_img"] += recall_per_img
                stats[cat_type]["cat_exists"] += gt_total_per_cat > 0
                stats[cat_type]["img_with_pos"] += 1

        # Return collection of recall values, along with their averages. For the latter,
        # Recall@K per image values should be divided by approporiate denominators (i.e.
        # whether certain category has occurred in each image or not), collected along
        # the recall values
        recalls_per_cat = {
            cat_type: {
                # Per-category recall@K values
                **{
                    f"{cat_type}_recall@{k}": rcs["per_cat"][:,i] / rcs["cat_exists"]
                    for i, k in enumerate(ks)
                }
            }
            for cat_type, rcs in stats.items()
        }
        recalls_avg = {
            cat_type: {
                # Per-category recall@K values
                **{
                    f"{cat_type}_recall_cat@{k}": float((
                        rcs["per_cat"][:,i] / rcs["cat_exists"]
                    )[rcs["cat_exists"] > 0].mean())
                    for i, k in enumerate(ks)
                },
                # Per-image recall@K values
                **{
                    f"{cat_type}_recall_img@{k}": float((
                        rcs["per_img"][i] / rcs["img_with_pos"]
                    ))
                    for i, k in enumerate(ks)
                }
            }
            for cat_type, rcs in stats.items()
        }

        recalls_per_cat = {
            "bbox": {
                **recalls_per_cat["cls"],
                **recalls_per_cat["att"],
                **recalls_per_cat["rel"]
            }
        }
        recalls_avg = {
            "bbox": {
                **recalls_avg["cls"],
                **recalls_avg["att"],
                **recalls_avg["rel"]
            }
        }

        return recalls_per_cat, recalls_avg


class VGFewShotEvaluator(DatasetEvaluator):
    """
    Evaluate few-shot category classification performance with the mean average precision
    (mAP) metric -- with AP not as in typical object detection task, but within each few-shot
    recognition episode. Contrary to _VGBatchEvaluators, annotations are provided along the
    predictions and don't have to be loaded from files, and model outputs are much more
    lightweight: thus considerably more straightforward to implement evaluation.
    """
    def __init__(self, dataset_name):
        self._dataset_name = dataset_name
        md = MetadataCatalog.get(self._dataset_name)

        self._predicates = {
            "cls": md.classes,
            "att": md.attributes,
            "rel": md.relations
        }

        self._cpu_device = torch.device("cpu")

    def reset(self):
        """ Prepare empty dict to store results """
        self._predictions = {
            "cls": [],
            "att": [],
            "rel": []
        }

    def process(self, inputs, outputs):
        """ Aggregating predictions + ground_truths """
        for _, output in zip(inputs, outputs):
            for cat_type, pred in output.items():
                self._predictions[cat_type].append(pred)

    def evaluate(self):
        """ Compute evaluation metrics """
        comm.synchronize()
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return {}, {}

        predictions = { "cls": [], "att": [], "rel": [] }
        for preds in all_predictions:
            for cat_type in preds:
                predictions[cat_type] += preds[cat_type]
        del all_predictions

        average_precisions = { "cls": {}, "att": {}, "rel": {} }
        for cat_type in predictions:
            for ep_result in predictions[cat_type]:
                scores = ep_result["scores"].to(self._cpu_device)
                ground_truth = ep_result["ground_truth"].to(self._cpu_device)

                sort_inds = np.argsort(-scores)

                # Compute precision-recall curve(s)
                pred_is_TP = torch.take_along_dim(ground_truth, sort_inds, dim=-1)
                TPs = pred_is_TP.cumsum(dim=-1)
                FPs = (1 - pred_is_TP).cumsum(dim=-1)
                precision = TPs / (TPs+FPs)
                recall = TPs / ground_truth.sum(dim=-1, keepdim=True)

                for i, c in enumerate(ep_result["categories"]):
                    average_precisions[cat_type][c] = _AP_from_PR(
                        precision[i,:], recall[i,:]
                    )

        metrics_per_cat = {
            "few_shot": {
                f"{cat_type}_AP": [
                    APs[c] if c in APs else None
                    for c in range(len(self._predicates[cat_type]))
                ]
                for cat_type, APs in average_precisions.items()
            }
        }
        metrics_avg = {
            "few_shot": {
                f"{cat_type}_mAP": np.mean([ap for ap in APs.values() if ap is not None])
                for cat_type, APs in average_precisions.items()
            }
        }
        
        return metrics_per_cat, metrics_avg


class _NullMax():
    """
    For passing around 'null' values of torch.return_types.max type for empty annos
    """
    def __init__(self, size):
        self.values = torch.zeros(size)
        self.indices = None     # This field is never actually used


def _AP_from_PR(precision, recall):
    """
    Calculation of AP from precision-recall curve, in PASCAL VOC (after 2008) style.
    Taken mostly from evaluation/pascal_voc_evaluation.py at detectron2 main repo.
    """
    if recall.isnan().sum() > 0:
        # AP is not defined when there's no ground-truth occurrence whatsoever
        return None
    
    else:
        # First append sentinel values at the end
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))

        # Compute the precision envelope
        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = np.maximum(precision[i - 1], precision[i])

        # To calculate area under PR curve, look for points where X axis (recall)
        # changes value
        i = np.where(recall[1:] != recall[:-1])[0]

        # ... and sum (\Delta recall) * prec
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

        return ap


def _approx_score(score):
    """
    Round given float score in [0, 1] to an int in [0, 10000], then return as string
    """
    return str(round(score*1e4))


def _1hot(indices, C):
    """
    Could use F.one_hot(), but this can directly create tensors with dtype=torch.int16)
    """
    return torch.eye(C, dtype=torch.int16)[indices]
