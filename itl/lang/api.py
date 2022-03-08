"""
Language processing module API that exposes only the high-level functionalities
required by the ITL agent: situate the embodied agent in a physical environment,
understand & generate language input in the context of the dialogue
"""
import numpy as np

from .semantics import SemanticParser
from .dialogue import DialogueManager


class LanguageModule:

    def __init__(self, opts, lex=None):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.opts = opts
        self.semantic = SemanticParser(opts.grammar_image_path, opts.ace_binary_path)
        self.dialogue = DialogueManager()
        self.lexicon = lex

        self.vis_raw = None

    def situate(self, vis_raw, vis_scene, objectness_thresh=0.5):
        """
        Put entities in the physical environment into domain of discourse
        """
        # Start a dialogue information state anew
        self.dialogue.refresh()

        # Store raw visual perception so that it can be used during 'pointing' gesture
        self.vis_raw = vis_raw

        # Filter by objectness threshold (recognized objects are already ranked by
        # objectness score)
        vis_scene = [
            obj for obj in vis_scene if obj["pred_objectness"] > objectness_thresh
        ]
        # Accordingly crop relation prediction matrices
        for obj in vis_scene:
            obj["pred_relations"] = obj["pred_relations"][:len(vis_scene)]

        # Incorporate parsed scene graph into dialogue context
        for oi, obj in enumerate(vis_scene):
            bbox = obj["pred_boxes"]
            self.dialogue.referents["env"][f"o{oi}"] = {
                "bbox": bbox,
                "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            }
        
        # Register these indices as names, for starters
        self.dialogue.referent_names = {i: i for i in self.dialogue.referents["env"]}

    def understand(self, usr_in):
        return self.dialogue.understand(usr_in, self.semantic, self.lexicon, self.vis_raw)

    def generate(self):
        raise NotImplementedError
