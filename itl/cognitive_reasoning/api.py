"""
Cognitive reasoning module API that exposes only the high-level functionalities
required by the ITL agent: make sense out of the current visual & language inputs
plus existing knowledge stored in knowledge base
"""
from .sensemake import sensemake_vis, sensemake_vis_lang


class CognitiveReasonerModule:

    def __init__(self, kb):
        self.kb = kb

    def sensemake(self, vis_scene, dialogue_state, lexicon):
        vis_out = sensemake_vis(vis_scene)
        if len(dialogue_state["record"]) > 0:
            vis_lang_out = sensemake_vis_lang(vis_out, dialogue_state, lexicon)
        else:
            vis_lang_out = None
        
        return vis_out, vis_lang_out
