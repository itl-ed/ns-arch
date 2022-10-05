"""
Implement custom mapper function for VG DataLoader
"""
import copy
import logging

import numpy as np

import torch
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from detectron2.data import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.config import configurable
from detectron2.structures import BoxMode, Instances, Boxes


logger = logging.getLogger("vision.data.mapper")

class _VGMapper(DatasetMapper):
    """
    To be extended by two subclasses, each for batch/few-shot learning mode; implement
    shared methods here, and implement __call__() differently for subclasses
    """

    def __call__(self, dataset_dict):
        raise NotImplementedError

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        """
        Override the method by replacing utils.annotations_to_instances with
        a custom method
        """
        ## Unused; w/o segmentation & keypoint annotations
        # for anno in dataset_dict["annotations"]:
        #     if not self.use_instance_mask:
        #         anno.pop("segmentation", None)
        #     if not self.use_keypoint:
        #         anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        # Using the different method here
        instances = self._vg_annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        ## Unused; w/o segmentation & keypoint annotations
        # if self.recompute_boxes:
        #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        instances, filtered = utils.filter_empty_instances(instances, return_mask=True)

        if self.gt_box_proposals:
            if all(["objectness_scores" in obj for obj in annos]):
                instances.pred_objectness = torch.stack([
                    torch.tensor(obj["objectness_scores"])
                        if obj["objectness_scores"] is not None
                        else torch.tensor([1.0])
                    for obj in annos
                ])
            dataset_dict["proposals"] = instances

            return
        else:
            dataset_dict["instances"] = instances

        # Need to make sure the second dims of gt_relations are filtered as well
        gt_rels_flt = dataset_dict["instances"].gt_relations[:, filtered, :]
        assert gt_rels_flt.shape[0] == gt_rels_flt.shape[1]

        dataset_dict["instances"].gt_relations = gt_rels_flt
    
    def _vg_annotations_to_instances(self, annos, image_size, mask_format="polygon"):
        """
        Custom extension of utils.annotations_to_instances, which can handle
        VG scene graph annotations (i.e. attributes & relations) in addition to
        (multi-label) object class annotations
        """
        boxes = (
            np.stack(
                [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
            )
            if len(annos)
            else np.zeros((0, 4))
        )
        target = Instances(image_size)
        target.gt_boxes = Boxes(boxes)

        # When ground truth boxes are needed as proposals
        if self.gt_box_proposals:
            target.proposal_boxes = target.gt_boxes
            if not self.is_train:
                return target        # Can return during inference

        assert hasattr(self, "num_preds"), "VG mapper doesn't have any predicate vocab size info"

        # Object classes: (N_objs, N_classes) sparse (one-hot) tensor
        classes = [[[o_i, o_c] for o_c in obj["classes"]] for o_i, obj in enumerate(annos)]
        classes = torch.tensor(sum(classes, []))
        if len(classes) > 0:
            classes = torch.sparse.LongTensor(
                classes.T, torch.ones(len(classes), dtype=torch.long), (len(annos), self.num_preds["cls"])
            ).to_dense()
        else:
            classes = torch.zeros([len(annos), self.num_preds["cls"]], dtype=torch.long)
        target.gt_classes = classes

        # (For few-shot training) Only needs up to gt_classes when self.gt_box_proposals
        if self.gt_box_proposals: return target

        # Object attributes: (N_objs, N_attributes) sparse (one-hot) tensor
        attributes = [[[o_i, o_a] for o_a in obj["attributes"]] for o_i, obj in enumerate(annos)]
        attributes = torch.tensor(sum(attributes, []))
        if len(attributes) > 0:
            attributes = torch.sparse.LongTensor(
                attributes.T, torch.ones(len(attributes), dtype=torch.long), (len(annos), self.num_preds["att"])
            ).to_dense()
        else:
            attributes = torch.zeros([len(annos), self.num_preds["att"]], dtype=torch.long)
        target.gt_attributes = attributes

        # Obj*Obj relations: (N_objs, N_objs, N_relations) sparse (one-hot) tensor
        rel_inds = {obj["object_id"]: oi for oi, obj in enumerate(annos)}  # Img-relative indices
        relations = [[sum([[o_i, rel_inds[o_r["object_id"]], r] for r in o_r["relation"]], [])
            for o_r in obj["relations"]] for o_i, obj in enumerate(annos)]
        relations = torch.tensor(sum(relations, []))
        if len(relations) > 0:
            relations = torch.sparse.LongTensor(
                relations.T, torch.ones(len(relations), dtype=torch.long), (len(annos), len(annos), self.num_preds["rel"])
            ).to_dense()
        else:
            relations = torch.zeros([len(annos), len(annos), self.num_preds["rel"]], dtype=torch.long)
        target.gt_relations = relations

        ## Unused; w/o segmentation & keypoint annotations
        # if len(annos) and "segmentation" in annos[0]:
        #     segms = [obj["segmentation"] for obj in annos]
        #     if mask_format == "polygon":
        #         try:
        #             masks = PolygonMasks(segms)
        #         except ValueError as e:
        #             raise ValueError(
        #                 "Failed to use mask_format=='polygon' from the given annotations!"
        #             ) from e
        #     else:
        #         assert mask_format == "bitmask", mask_format
        #         masks = []
        #         for segm in segms:
        #             if isinstance(segm, list):
        #                 # polygon
        #                 masks.append(polygons_to_bitmask(segm, *image_size))
        #             elif isinstance(segm, dict):
        #                 # COCO RLE
        #                 masks.append(mask_util.decode(segm))
        #             elif isinstance(segm, np.ndarray):
        #                 assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
        #                     segm.ndim
        #                 )
        #                 # mask array
        #                 masks.append(segm)
        #             else:
        #                 raise ValueError(
        #                     "Cannot convert segmentation of type '{}' to BitMasks!"
        #                     "Supported types are: polygons as list[list[float] or ndarray],"
        #                     " COCO-style RLE as a dict, or a binary segmentation mask "
        #                     " in a 2D numpy array of shape HxW.".format(type(segm))
        #                 )
        #         # torch.from_numpy does not support array with negative stride.
        #         masks = BitMasks(
        #             torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        #         )
        #     target.gt_masks = masks

        # if len(annos) and "keypoints" in annos[0]:
        #     kpts = [obj.get("keypoints", []) for obj in annos]
        #     target.gt_keypoints = Keypoints(kpts)

        return target


class VGBatchMapper(_VGMapper):
    """For batch-training scene graph generation model with Visual Genome dataset"""

    @configurable
    def __init__(
        self,
        is_train,
        *,
        provide_proposals=False,
        **kwargs
    ):
        """
        Option to provide ground-truth object bboxes as proposals, essentially making
        the model to solve classification tasks
        """
        super().__init__(is_train, **kwargs)
        self.gt_box_proposals = provide_proposals

    def __call__(self, dataset_dict):
        """
        Mostly identical to the original method, except overridden to allow annotations
        to pass if self.gt_box_proposals is True, and provide the transformed annotations
        as 'proposals' (gt_boxes only)
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        if "file_name" in dataset_dict:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        else:
            assert "image" in dataset_dict, "Must provide image data if 'file_name' not given"
            image = dataset_dict["image"]

        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError:
            # It really feels guilty to do this...
            logger.debug(f"Size mismatch happened with f{dataset_dict['file_name']}")

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # MODIFIED
        if not self.is_train and not self.gt_box_proposals:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        # MODIFIED

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class VGFewShotMapper(_VGMapper):
    """For few-shot-training scene graph generation model with Visual Genome dataset"""

    @configurable
    def __init__(
        self,
        is_train,
        **kwargs
    ):
        """
        Option to provide ground-truth object bboxes as proposals, essentially making
        the model to solve classification tasks
        """
        super().__init__(is_train, **kwargs)
        self.gt_box_proposals = True     # Always True in this mapper

    def __call__(self, dataset_dict):
        """
        Rather more deviant from the original method; here, dataset_dict is a dict containing
        sampled few-shot episodes for each category type (cls, att, rel).
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        K = dataset_dict.pop("shot", None)

        # In total, body of this triple loop runs 3 * N * I times
        for episode in dataset_dict.values():
            for sampled_instances in episode.values():
                for ins in sampled_instances:
                    self._process_instance(ins)
        
        if K is not None: dataset_dict["shot"] = K

        return dataset_dict

    def _process_instance(self, ins):
        """ Helper method factored out due to too much indentation :p """
        image = utils.read_image(ins["file_name"], format=self.image_format)

        try:
            # Sort of abusing the check_image_size() arg signature, by passing a
            # dict of image info instead of the default expectation of detectron2
            # dataset_dict
            utils.check_image_size(ins, image)
        except SizeMismatchError:
            logger.debug(f"Size mismatch happened with f{ins['file_name']}")

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        ins["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "annotations" in ins:
            self._transform_annotations(ins, transforms, image_shape)
