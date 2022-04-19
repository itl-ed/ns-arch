"""
Language processing module API that exposes only the high-level functionalities
requtt_idred by the ITL agent: situate the embodied agent in a physical environment,
understand & generate language input in the context of the dialogue
"""
from .semantics import SemanticParser
from .dialogue import DialogueManager
from ..lpmln import Literal, Rule
from ..lpmln.utils import wrap_args


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

    def understand(self, usr_in, agenda):
        self.dialogue.understand(usr_in, self.semantic, self.lexicon, self.vis_raw, agenda)

    def generate(self):
        """ Flush the buffer of utterances prepared """
        while len(self.dialogue.to_generate) > 0:
            utt = self.dialogue.to_generate.pop()
            print(f"A> {utt}")
    
    def export_dialogue_state(self):
        """ Export the current dialogue information state as a dict """
        return vars(self.dialogue)
    
    def update_referent_assignment(self, assignment):
        """
        Update assignment from discourse referents to environment referents with provided mapping
        (Overwrite if key exists)
        """
        for dis_ref, env_ref in assignment.items():
            self.dialogue.assignment_soft[dis_ref] = env_ref

    def utt_contains_neologism(self, utt_id):
        """
        Check if logical content of dialogue record indexed by utt_id contains any neologism
        """
        _, _, (rules, queries), _ = self.dialogue.record[utt_id]

        if rules is not None:
            for head, body, _ in rules:
                if head is not None:
                    if head[:2] not in self.lexicon.s2d: return True
                if body is not None:
                    for b in body:
                        if b[:2] not in self.lexicon.s2d: return True
        
        if queries is not None:
            for _, q_lits in queries:
                for ql in q_lits:
                    if ql[:2] not in self.lexicon.s2d: return True

        return False

    def utt_to_ASP(self, utt_id):
        """
        Convert logical content of dialogue record (which should be already ASP-compatible)
        indexed by utt_id to ASP (LP^MLN) rule(s) and return
        """
        utt = self.dialogue.record[utt_id]
        _, _, (rules, queries), orig_utt = utt

        assig = {**self.dialogue.assignment_soft, **self.dialogue.assignment_hard}

        if rules is not None:
            converted_rules = []
            for head, body, _ in rules:
                if head is not None:
                    cat_ind, cat_type = self.lexicon.s2d[head[:2]]
                    wrapped_head = Literal(
                        f"{cat_type}_{cat_ind}", wrap_args(*[assig[a] for a in head[2]])
                    )
                else:
                    wrapped_head = None

                wrapped_bls = []
                if body is not None:
                    for bl in body:
                        cat_ind, cat_type = self.lexicon.s2d[bl[:2]]
                        bl = Literal(
                            f"{cat_type}_{cat_ind}", wrap_args(*[assig[a] for a in bl[2]])
                        )
                        wrapped_bls.append(bl)
                
                r = Rule(head=wrapped_head, body=wrapped_bls)
                converted_rules.append((r, orig_utt))
        else:
            converted_rules = None
        
        if queries is not None:
            converted_queries = []
            for q_ent, q_lits in queries:
                wrapped_qls = []
                for ql in q_lits:
                    cat_ind, cat_type = self.lexicon.s2d[ql[:2]]
                    ql = Literal(
                        f"{cat_type}_{cat_ind}", wrap_args(*[assig[a] for a in ql[2]])
                    )
                    wrapped_qls.append(ql)
                converted_queries.append((q_ent, wrapped_qls, orig_utt))
        else:
            converted_queries = None

        return converted_rules, converted_queries

    def prepare_answer(self, utt_id, cognitive):
        """ Synthesize natural language answer to question """
        self.dialogue.to_generate.append(f"Answer: {cognitive.Q_answers[utt_id]}")
