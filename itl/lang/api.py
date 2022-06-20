"""
Language processing module API that exposes only the high-level functionalities
requtt_idred by the ITL agent: situate the embodied agent in a physical environment,
understand & generate language input in the context of the dialogue
"""
from .semantics import SemanticParser
from .dialogue import DialogueManager


class LanguageModule:

    def __init__(self, opts):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.opts = opts
        self.semantic = SemanticParser(opts.grammar_image_path, opts.ace_binary_path)
        self.dialogue = DialogueManager()

        self.vis_raw = None
        self.unresolved_neologisms = set()

    def situate(self, vis_raw, vis_scene):
        """
        Put entities in the physical environment into domain of discourse
        """
        # No-op if no new visual input
        if vis_raw is None and vis_scene is None:
            return

        # Start a dialogue information state anew
        self.dialogue.refresh()

        # Store raw visual perception so that it can be used during 'pointing' gesture
        self.vis_raw = vis_raw

        # Incorporate parsed scene graph into dialogue context
        for oi, obj in vis_scene.items():
            bbox = obj["pred_boxes"]
            self.dialogue.referents["env"][oi] = {
                "bbox": bbox,
                "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            }
        
        # Register these indices as names, for starters
        self.dialogue.referent_names = {i: i for i in self.dialogue.referents["env"]}

    def understand(self, usr_in, vis_raw):
        self.dialogue.understand(usr_in, self.semantic, vis_raw)

    def generate(self):
        """ Flush the buffer of utterances prepared """
        while len(self.dialogue.to_generate) > 0:
            utt = self.dialogue.to_generate.pop()
            print(f"A> {utt}")
