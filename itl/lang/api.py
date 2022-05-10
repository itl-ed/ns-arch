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
        self.dialogue.assignment_soft.update(assignment)
    
    def update_word_senses(self, word_senses):
        """
        Update word sense denotation under current dialogue context. Currently symbol type based,
        but sophisticated use cases will require token based sense resolution.
        """
        self.dialogue.word_senses.update(word_senses)

    def utt_contains_neologism(self, utt_id):
        """
        Check if logical content of dialogue record indexed by utt_id contains any neologism
        """
        _, _, (rules, query), _ = self.dialogue.record[utt_id]

        if rules is not None:
            for head, body, _ in rules:
                if head is not None:
                    if head[:2] not in self.lexicon.s2d: return True
                if body is not None:
                    for b in body:
                        if b[:2] not in self.lexicon.s2d: return True
        
        if query is not None:
            _, q_rules = query
            for head, body, _ in q_rules:
                if head is not None:
                    if head[:2] not in self.lexicon.s2d: return True
                if body is not None:
                    for b in body:
                        if b[:2] not in self.lexicon.s2d: return True

        return False

    def utt_to_ASP(self, utt_id):
        """
        Convert logical content of dialogue record (which should be already ASP-compatible)
        indexed by utt_id to ASP (LP^MLN) rule(s) and return
        """
        utt = self.dialogue.record[utt_id]
        _, _, (rules, query), orig_utt = utt

        assig = {**self.dialogue.assignment_soft, **self.dialogue.assignment_hard}

        if rules is not None:
            converted_rules = []
            for head, body, _ in rules:
                if head is not None:
                    cat_ind, cat_type = self.dialogue.word_senses[head[:2]]
                    wrapped_head = Literal(
                        f"{cat_type}_{cat_ind}", wrap_args(*[assig[a] for a in head[2]])
                    )
                else:
                    wrapped_head = None

                wrapped_bls = []
                if body is not None:
                    for bl in body:
                        cat_ind, cat_type = self.dialogue.word_senses[bl[:2]]
                        bl = Literal(
                            f"{cat_type}_{cat_ind}", wrap_args(*[assig[a] for a in bl[2]])
                        )
                        wrapped_bls.append(bl)
                
                r = Rule(head=wrapped_head, body=wrapped_bls)
                converted_rules.append((r, orig_utt))
        else:
            converted_rules = None

        if query is not None:
            q_ents, q_rules = query

            wrapped_qrs = []
            for head, body, _ in q_rules:
                if head is not None:
                    # Here we are resolving any homonymy by always choosing denotation with
                    # the highest frequency. In distant future we may implement exploiting
                    # discourse/environment contexts for word sense disambiguation here...?
                    cat_ind, cat_type = max(
                        self.lexicon.s2d[head[:2]], key=lambda d: self.lexicon.d_freq[d]
                    )
                    wrapped_head = Literal(
                        f"{cat_type}_{cat_ind}",
                        wrap_args(*[assig.get(a, a) for a in head[2]])
                    )
                else:
                    wrapped_head = None

                wrapped_bls = []
                if body is not None:
                    for bl in body:
                        cat_ind, cat_type = max(
                            self.lexicon.s2d[head[:2]], key=lambda d: self.lexicon.d_freq[d]
                        )
                        bl = Literal(
                            f"{cat_type}_{cat_ind}",
                            wrap_args(*[assig.get(a, a) for a in bl[2]])
                        )
                        wrapped_bls.append(bl)

                r = Rule(head=wrapped_head, body=wrapped_bls)
                wrapped_qrs.append(r)

            converted_query = (q_ents, wrapped_qrs, orig_utt)
        else:
            converted_query = None

        return converted_rules, converted_query

    def prepare_answer(self, utt_id, cognitive):
        """ Synthesize natural language answer to question """
        self.dialogue.to_generate.append(f"Answer: {cognitive.Q_answers[utt_id]}")
