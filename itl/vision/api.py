"""
Vision processing module API that exposes only the high-level functionalities
required by the ITL agent inference (full scene graph generation, classification
given bbox and visual search by concept exemplars). Implemented using publicly
released model of OWL-ViT.
"""
from PIL import Image

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class VisionModule:

    def __init__(self, model_path):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.scene = None
        self.last_input = None
        self.last_raw = None

        # Inventory of distinct visual concepts that the module (and thus the agent
        # equipped with this module) is aware of, one per concept category. Right now
        # I cannot think of any specific type of information that has to be stored in
        # this module (exemplars are stored in long term memory), so let's just keep
        # only integer sizes of inventories for now...
        self.inventories = { "cls": 0, "att": 0, "rel": 0 }

        # Load a pre-trained OwL-ViT processor & model from path (huggingface hub
        # model id or local path) provided in opts
        cache_dir = "./assets/vision_models/"
        self.processor = OwlViTProcessor.from_pretrained(
            model_path, cache_dir=cache_dir
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            model_path, cache_dir=cache_dir
        )
        if torch.cuda.is_available():
            self.model.to("cuda")

    def owlvit_process(self, image, label_texts=None, label_exemplars=None):
        """
        Process image using OwL-ViT model, with label space defined by either
        label_texts or label_exemplars (but not both).
        """
        # Must provide either set of label texts, or set of exemplars of concepts
        assert label_texts is not None or label_exemplars is not None

        if isinstance(image, str):
            # Read image data from path
            image = Image.open(image)

        # Process input image (optionally along with label texts that is not None)
        # prepare OwL-ViT input
        input = self.processor(images=image, text=label_texts, return_tensors="pt")
        if torch.cuda.is_available():
            for k, v in input.data.items():
                input.data[k] = v.to("cuda")

        # Feed processed input to model to obtain output
        output = self.model(**input)

        # Post-process model output into intelligible formats
        image_size = torch.Tensor([image.size[::-1]])
        results = self.post_process(outputs=output, target_sizes=image_size)

        return results, output, image

    def add_concept(self, cat_type):
        """
        Register a novel visual concept to the model, expanding the concept inventory of
        corresponding category type (class/attribute/relation). Note that visual concepts
        are not inseparably attached to some linguistic symbols; such connections are rather
        incidental and should be established independently (consider synonyms, homonyms).
        Plus, this should allow more flexibility for, say, multilingual agents, though there
        is no plan to address that for now...

        Returns the index of the newly added concept.
        """
        self.inventories[cat_type] += 1
        return self.inventories[cat_type]

    def predict(self, image, label_texts=None, label_exemplars=None, bboxes=None, specs=None):
        """
        Model inference in either one of three modes:
            1) full scene graph generation mode, where the module is only given an image
                and needs to return its estimation of the full scene graph for the input
            2) instance classification mode, where a number of bboxes are given along
                with the image and category predictions are made for only those instances
            3) instance search mode, where a specification is provided in the form of FOL
                formula with a variable and best fitting instance(s) should be searched

        2) and 3) are 'incremental' in the sense that they should add to an existing scene
        graph which is already generated with some previous execution of this method. Provide
        bboxes arg to run in 2) mode, or spec arg to run in 3) mode.
        """
        # Must provide either set of label texts, or set of exemplars of concepts
        assert label_texts is not None or label_exemplars is not None

        # Prediction modes
        if bboxes is None and specs is None:
            # Full (ensemble) prediction
            results, output, input_img = self.owlvit_process(
                image, label_texts=label_texts, label_exemplars=label_exemplars
            )
            self.last_input = input_img

            print(0)

        else:
            if bboxes is not None:
                # Instance classification mode
                print(0)
            else:
                assert specs is not None
                # Instance search mode
                print(0)

    def post_process(self, outputs, target_sizes):
        """
        Basically identical to OwlViTProcessor().post_process, except it is ensured
        all intermediate tensors are on the same device.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = _center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(device=boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


def _center_to_corners_format(x):
    """
    Helper method in OwlViTFeatureExtractor module exposed for access
    """
    x_center, y_center, width, height = x.unbind(-1)
    boxes = [(x_center - 0.5 * width), (y_center - 0.5 * height), (x_center + 0.5 * width), (y_center + 0.5 * height)]
    return torch.stack(boxes, dim=-1)
