"""
Implement custom modules by extending detectron2-provided defaults, to be plugged
into our scene graph generation model
"""
import torch
import numpy as np
from detectron2.modeling import (
    GeneralizedRCNN,
    META_ARCH_REGISTRY
)
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class DualModeRCNN(GeneralizedRCNN):
    """
    GeneralizedRCNN extended to accept inputs with already prepared proposal boxes;
    for such inputs self.proposal_generator is suppressed, and RoI heads are applied
    on the provided proposals only, in effect solving classification problem
    """
    def inference(
        self, batched_inputs,
        detected_instances=None, do_postprocess=True, exs_cached=None, search_specs=None
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        if exs_cached is None:
            features = self.backbone(images.tensor)
        else:
            # Use cached backbone output
            features = exs_cached["backbone_output"]

        if detected_instances is None:
            if "proposals" in batched_inputs[0]:
                # When "proposals" already present in input, treat them as ground-truth;
                # run roi_heads on the boxes with boxes_provided=True to prevent any of them
                # getting filtered out
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                results, f_vecs, inc_out = self.roi_heads(
                    images, features, proposals, targets=None,
                    exs_cached=exs_cached, boxes_provided=True
                )
            else:
                # Otherwise, default behavior of the parent class
                assert self.proposal_generator is not None
                proposals, _ = self.proposal_generator(images, features, None)
                results, f_vecs, inc_out = self.roi_heads(
                    images, features, proposals, targets=None,
                    exs_cached=exs_cached, boxes_provided=False, search_specs=search_specs
                )
        else:
            # No use case handled by this snippet yet...
            raise NotImplementedError
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            f_vecs = None

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self._postprocess(results, batched_inputs, images.image_sizes), f_vecs, inc_out
        else:
            return results, f_vecs, inc_out

    def visualize_training(self, batched_inputs, proposals):
        """
        Minimal modification for moving input["instances"].gt_boxes to cpu device before
        v_gt.overlay_instances(), and *NOT* transposing image dimensions
        """
        from detectron2.utils import comm
        from detectron2.utils.visualizer import Visualizer

        if not comm.is_main_process():
            return

        storage = get_event_storage()
        max_vis_prop = 30

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes.tensor.cpu())
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
