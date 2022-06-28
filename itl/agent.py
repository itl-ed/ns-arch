"""
Outermost wrapper containing ITL agent API
"""
import math
import readline
import rlcompleter

import numpy as np
from detectron2.structures import BoxMode

from .memory import LongTermMemoryModule
from .vision import VisionModule
from .vision.utils.completer import DatasetImgsCompleter
from .lang import LanguageModule
from .recognitive_reasoning import RecognitiveReasonerModule
from .practical_reasoning import PracticalReasonerModule
from .lpmln import Rule, Literal
from .lpmln.utils import wrap_args


# FT_THRES = 0.5              # Few-shot learning trigger score threshold
SR_THRES = -math.log(0.5)   # Mismatch surprisal threshold
U_W_PR = 1.0                # How much the agent values information provided by the user
EPS = 1e-10                 # Value used for numerical stabilization
TAB = "\t"                  # For use in format strings

class ITLAgent:

    def __init__(self, opts):
        self.opts = opts

        # Initialize component modules
        self.lt_mem = LongTermMemoryModule()
        self.vision = VisionModule(opts)
        self.lang = LanguageModule(opts)
        self.recognitive = RecognitiveReasonerModule()
        self.practical = PracticalReasonerModule()

        # Initialize empty lexicon with concepts in visual module
        self.lt_mem.lexicon.fill_from_dicts(
            self.vision.predicates, self.vision.predicates_freq
        )

        # Show visual UI and plots
        self.vis_ui_on = True

        # Image file selection CUI
        self.dcompleter = DatasetImgsCompleter()
        readline.parse_and_bind("tab: complete")

    def __call__(self):
        """Main function: Kickstart infinite ITL agent loop with user interface"""
        print(f"Sys> At any point, enter 'exit' to quit")

        while True:
            self.loop()
    
    def loop(self, v_usr_in=None, l_usr_in=None, pointing=None):
        """
        Single agent activity loop. Provide usr_in for programmatic execution; otherwise,
        prompt user input on command line REPL
        """
        self._vis_inp(usr_in=v_usr_in)
        self._lang_inp(usr_in=l_usr_in)
        self._update_belief(pointing=pointing)
        act_out = self._act()

        return act_out

    def _vis_inp(self, usr_in=None):
        """Image input prompt (Choosing from dataset for now)"""
        self.vision.new_input = None
        input_provided = usr_in is not None

        # Register autocompleter for REPL
        readline.set_completer(self.dcompleter.complete)

        print("")

        while True:

            print(f"Sys> Choose an image to process")
            if input_provided:
                print(f"U> {usr_in}")
            else:
                print("Sys> Enter 'r' for random selection, 'n' for skipping new image input")
                usr_in = input("U> ")
                print("")

            try:
                if usr_in == "n":
                    print(f"Sys> Skipped image selection")
                    break
                elif usr_in == "r":
                    self.vision.new_input = self.dcompleter.sample()
                    self.vision.last_bboxes = None
                elif usr_in == "exit":
                    print(f"Sys> Terminating...")
                    quit()
                else:
                    if usr_in not in self.dcompleter:
                        raise ValueError(f"Image file {usr_in} does not exist")
                    self.vision.new_input = usr_in
                    self.vision.last_bboxes = None

            except ValueError as e:
                if input_provided:
                    raise e
                else:
                    print(f"Sys> {e}, try again")

            else:
                self.vision.last_input = self.vision.new_input
                print(f"Sys> Selected image file: {self.vision.new_input}")
                break

        # Restore default completer
        readline.set_completer(rlcompleter.Completer().complete)
    
    def _lang_inp(self, usr_in=None):
        """Language input prompt (from user)"""
        self.lang.new_input = None
        input_provided = usr_in is not None

        print("Sys> Awaiting user input...")
        print("Sys> Enter 'n' for skipping language input")

        while True:
            if input_provided:
                print(f"U> {usr_in}")
            else:
                usr_in = input("U> ")
                print("")

            try:
                if usr_in == "n":
                    print(f"Sys> Skipped language input")
                    break
                elif usr_in == "exit":
                    print(f"Sys> Terminating...")
                    quit()
                elif usr_in == "Correct.":
                    # Positive feedback provided
                    break
                else:
                    self.lang.new_input = self.lang.semantic.nl_parse(usr_in)
                    break

            except IndexError as e:
                if input_provided:
                    raise e
                else:
                    print(f"Sys> Ungrammatical input or IndexError: {e.args}")
            except ValueError as e:
                if input_provided:
                    raise e
                else:
                    print(f"Sys> {e.args[0]}")

    def _update_belief(self, pointing=None):
        """ Form beliefs based on visual and/or language input """

        if not (self.vision.new_input or self.lang.new_input):
            # No information whatsoever to make any belief updates
            print("A> (Idling the moment away...)")
            return

        # Lasting storage of pointing info
        if pointing is None:
            pointing = {}
        
        # For showing visual UI on only the first time
        vis_ui_on = self.vis_ui_on

        # Keep updating beliefs until there's no more immediately exploitable learning
        # opportunities
        vision_model_updated = False    # Whether learning happened at neural-level
        kb_updated = False              # Whether learning happened at symbolic-level
        while True:
            ###################################################################
            ##                  Processing perceived inputs                  ##
            ###################################################################

            if self.vision.new_input is not None or vision_model_updated:
                # Ground raw visual perception with scene graph generation module
                self.vision.predict(
                    self.vision.last_input,
                    bboxes=self.vision.last_bboxes, visualize=vis_ui_on
                )
                vis_ui_on = False

            if self.vision.new_input is not None:
                # Inform the language module of the visual context
                self.lang.situate(self.vision.last_input, self.vision.scene)
                self.recognitive.refresh()

                # No need to treat these facts as 'mismatches'
                # (In a sense, this kinda overlaps with the notion of 'common ground'?
                # May consider fulfilling this later)
                self.doubt_no_more = set()

            # New index of utterance to be added to discourse record, if ever
            ui = len(self.lang.dialogue.record)

            # Understand the user input in the context of the dialogue
            if self.lang.new_input is not None:
                self.lang.understand(
                    self.lang.new_input, self.vision.last_input, ui=ui, pointing=pointing
                )

            if self.vision.scene is not None:
                # If some discourse referent is hard-assigned to some entity, boost its
                # objectness score so that it's captured during sensemaking
                for ent in self.lang.dialogue.assignment_hard.values():
                    if ent in self.vision.scene:
                        self.vision.scene[ent]["pred_objectness"] = np.array([1.0])

                # If a new entity is registered as a result of understanding the latest
                # input, re-run vision module to update with new predictions for it
                if len(self.lang.dialogue.referents["env"]) > len(self.vision.scene):
                    bboxes = [
                        {
                            "bbox": ent["bbox"],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "objectness_scores": self.vision.scene[name]["pred_objectness"]
                                if name in self.vision.scene else None
                        }
                        for name, ent in self.lang.dialogue.referents["env"].items()
                    ]

                    # Predict on latest raw data stored
                    self.vision.predict(self.vision.last_input, bboxes=bboxes)

            ###################################################################
            ##       Sensemaking via synthesis of perception+knowledge       ##
            ###################################################################

            dialogue_state = self.lang.dialogue.export_as_dict()
            kb_prog = self.lt_mem.kb.export_as_program()

            if self.vision.new_input is not None or vision_model_updated or kb_updated:
                # Sensemaking from vision input only
                self.recognitive.sensemake_vis(self.vision.scene, kb_prog)

            if self.lang.new_input is not None:
                # Reference & word sense resolution to connect vision & discourse
                self.recognitive.resolve_symbol_semantics(
                    dialogue_state, self.lt_mem.lexicon
                )

                if self.vision.scene is not None:
                    # Sensemaking from vision & language input
                    self.recognitive.sensemake_vis_lang(dialogue_state)

            ###################################################################
            ##           Identify & exploit learning opportunities           ##
            ###################################################################

            # Resetting flags
            vision_model_updated = False
            kb_updated = False

            # Process translated dialogue record to do the following:
            #   - Integrate newly provided generic rules into KB
            #   - Identify recognition mismatch btw. user provided vs. agent
            translated = self.recognitive.translate_dialogue_content(dialogue_state)
            for ui, (rules, _) in enumerate(translated):
                if rules is not None:
                    for r in rules:
                        # Symbolic knowledge base expansion; for generic rules without
                        # constant terms
                        if all(is_var for _, is_var in r.terms()):
                            # Integrate the rule into KB by adding (for now we won't
                            # worry about intra-KB consistency, etc.)
                            provenance = dialogue_state["record"][ui][3]
                            kb_updated |= self.lt_mem.kb.add(r, U_W_PR, provenance)

                        # Test against vision-only sensemaking result to identify any
                        # recognitive mismatch
                        if all(not is_var for _, is_var in r.terms()) \
                            and self.recognitive.concl_vis is not None:
                            if r in self.doubt_no_more:
                                # May skip testing this one
                                continue

                            # Make a yes/no query to obtain the likelihood of content
                            models_v, _, _ = self.recognitive.concl_vis
                            q_response, _ = models_v.query(None, r)
                            ev_prob = q_response[()][1]

                            surprisal = -math.log(ev_prob + EPS)
                            if surprisal > SR_THRES:
                                m = (r, surprisal)
                                self.recognitive.mismatches.add(m)

            # Handle neologisms
            neologisms = {
                tok: sym for tok, (sym, den) in self.recognitive.word_senses.items()
                if den is None
            }
            for tok, sym in neologisms.items():
                neo_in_rule_head = tok[1].startswith("r") and tok[2].startswith("h")
                neos_in_same_rule_body = [
                    n for n in neologisms if tok[:2]==n[:2] and n[2].startswith("b")
                ]
                if neo_in_rule_head and len(neos_in_same_rule_body)==0:
                    # Occurrence in rule head implies either definition or exemplar is
                    # provided by the utterance containing this token... Register new
                    # visual concept, and perform few-shot learning if appropriate
                    pos, name = sym
                    if pos == "n":
                        cat_type = "cls"
                    elif pos == "a":
                        cat_type = "att"
                    else:
                        assert pos == "v" or pos == "r"
                        cat_type = "rel"

                    # Expand corresponding visual concept inventory
                    concept_ind = self.vision.add_concept(cat_type)
                    novel_concept = (cat_type, concept_ind)

                    # Acquire novel concept by updating lexicon (and vision.predicates)
                    self.lt_mem.lexicon.add((name, pos), novel_concept)
                    self.vision.predicates[cat_type].append(
                        f"{name}.{pos}.0{len(self.lt_mem.lexicon.s2d[(name, pos)])}"
                    )

                    ui = int(tok[0].strip("u"))
                    ri = int(tok[1].strip("r"))
                    rule_head, rule_body, _ = dialogue_state["record"][ui][2][0][ri]

                    if rule_body is None:
                        # Labelled exemplar provided; add new concept exemplars to
                        # memory, as feature vectors at the penultimate layer right
                        # before category prediction heads
                        args = [
                            self.recognitive.value_assignment[arg] for arg in rule_head[0][2]
                        ]
                        if cat_type == "cls":
                            f_vec = self.vision.f_vecs[0][args[0]]
                        elif cat_type == "att":
                            f_vec = self.vision.f_vecs[1][args[0]]
                        else:
                            assert cat_type == "rel"
                            f_vec = self.vision.f_vecs[2][args[0]][args[1]]

                        self.lt_mem.exemplars.add_pos(novel_concept, f_vec.cpu().numpy())

                        # Update the category code parameter in the vision model's predictor
                        # head using the new set of exemplars
                        self.vision.update_concept(
                            novel_concept, self.lt_mem.exemplars[novel_concept], mix_ratio=1.0
                        )
                        vision_model_updated = True

                        # This now shouldn't strike the agent as surprise, at least in this
                        # loop (Ideally this doesn't need to be enforced this way, if the
                        # few-shot learning capability is perfect)
                        self.doubt_no_more.add(Rule(
                            head=Literal(f"{cat_type}_{concept_ind}", wrap_args(*args))
                        ))
                else:
                    # Otherwise not immediately resolvable
                    self.lang.unresolved_neologisms.add((sym, tok))

            # Terminate the loop when 'equilibrium' is reached
            if not (vision_model_updated or kb_updated):
                break

        # self.vision.reshow_pred()

    def _act(self):
        """
        Just eagerly try to resolve each item in agenda as much as possible, generating
        and performing actions until no more agenda items can be resolved for now. I
        wonder if we'll ever need a more sophisticated mechanism than this simple, greedy
        method for a good while?
        """
        ## Generate agenda items from maintenance goals
        # Currently, the maintenance goals are not to leave:
        #   - any unaddressed neologism which is unresolvable
        #   - any unaddressed recognition inconsistency btw. agent and user
        #   - any unanswered question that is answerable
        #
        # Ideally, this is to be accomplished declaratively by properly setting up formal
        # maintenance goals and then performing automated planning or something to come
        # up with right sequence of actions to be added to agenda. However, the ad-hoc code
        # below (+ plan library in practical/plans/library.py) will do for our purpose right
        # now; we will see later if we'll ever need to generalize and implement the said
        # procedure.)

        for n in self.lang.unresolved_neologisms:
            self.practical.agenda.append(("address_neologism", n))
        for m in self.recognitive.mismatches:
            self.practical.agenda.append(("address_mismatch", m))
        for ui in self.lang.dialogue.unanswered_Q:
            self.practical.agenda.append(("address_unanswered_Q", ui))

        return_val = []

        if self.lang.new_input is not None and len(self.practical.agenda) == 0:
            # Everything seems okay, acknowledge user input
            self.practical.agenda.append(("acknowledge", None))

        while True:
            resolved_items = []
            for i, todo in enumerate(self.practical.agenda):
                todo_state, todo_args = todo

                # Check if this item can be resolved at this stage and if so, obtain
                # appropriate plan (sequence of actions) for resolving the item
                plan = self.practical.obtain_plan(todo_state)

                if plan is not None:
                    # Perform plan actions
                    if plan is not None:
                        for action in plan:
                            act_method = action["action_method"].extract(self)
                            act_args = action["action_args_getter"](todo_args)
                            if type(act_args) == tuple:
                                act_args = tuple(a.extract(self) for a in act_args)
                            else:
                                act_args = (act_args.extract(self),)

                            act_out = act_method(*act_args)
                            if act_out is not None:
                                return_val.append(act_out)

                    resolved_items.append(i)

            if len(resolved_items) == 0:
                # No resolvable agenda item any more; break
                break
            else:
                # Check off resolved agenda item
                resolved_items.reverse()
                for i in resolved_items:
                    del self.practical.agenda[i]

        return return_val

    def handle_mismatch(self, mismatch):
        """
        Handle recognition gap following some specified strategy. Note that we now
        assume the user (teacher) is an infallible oracle, and the agent doesn't
        question info provided from user.
        """
        # This mismatch is about to be handled
        self.recognitive.mismatches.remove(mismatch)

        rule, _ = mismatch

        if self.opts.strat_mismatch == "zero_init":
            # Zero initiative from agent's end; do not ask any further question, simply
            # perform few-shot vision model updates (if possible) and acknowledge "OK"
            if rule.is_grounded() and len(rule.literals())==1:
                if rule.is_fact():
                    # Positive grounded fact
                    atom = rule.head[0]
                    exemplar_add_func = self.lt_mem.exemplars.add_pos
                else:
                    # Negative grounded fact
                    atom = rule.body[0]
                    exemplar_add_func = self.lt_mem.exemplars.add_neg

                cat_type, cat_ind = atom.name.split("_")
                cat_ind = int(cat_ind)
                args = [a for a, _ in atom.args]

                # Fetch current score for the asserted fact
                if cat_type == "cls":
                    f_vec = self.vision.f_vecs[0][args[0]]
                elif cat_type == "att":
                    f_vec = self.vision.f_vecs[1][args[0]]
                else:
                    assert cat_type == "rel"
                    f_vec = self.vision.f_vecs[2][args[0]][args[1]]

                imperfect_concept = (cat_type, cat_ind)

                # Add new concept exemplars to memory, as feature vectors at the
                # penultimate layer right before category prediction heads
                exemplar_add_func(imperfect_concept, f_vec.cpu().numpy())

                # Update the category code parameter in the vision model's predictor
                # head using the new set of exemplars
                self.vision.update_concept(
                    imperfect_concept, self.lt_mem.exemplars[imperfect_concept]
                )

                # This now shouldn't strike the agent as surprise, at least in this
                # loop (Ideally this doesn't need to be enforced this way, if the
                # few-shot learning capability is perfect)
                self.doubt_no_more.add(rule)

            self.practical.agenda.append(("acknowledge", None))

        elif self.opts.strat_mismatch == "request_exmp":
            raise NotImplementedError

        else:
            assert self.opts.strat_mismatch == "request_expl"
            raise NotImplementedError

    def attempt_answer_Q(self, ui):
        """
        Attempt to answer an unanswered question from user.
        
        If it turns out the question cannot be answered at all with the agent's current
        knowledge (e.g. question contains unresolved neologism), do nothing and wait for
        it to become answerable.

        If the agent can come up with an answer to the question, right or wrong, schedule
        to actually answer it by adding a new agenda item.
        """
        dialogue_state = self.lang.dialogue.export_as_dict()
        translated = self.recognitive.translate_dialogue_content(dialogue_state)

        _, query = translated[ui]

        if query is None:
            # Question cannot be answered for some reason
            return
        else:
            # Schedule to answer the question
            self.practical.agenda.append(("answer_Q", ui))
            return
    
    def prepare_answer_Q(self, ui):
        """
        Prepare an answer a question that has been deemed answerable, by first computing
        raw ingredients from which answer candidates can be composed, picking out an answer
        among the candidates, then translating the answer into natural language form to be
        uttered
        """
        # The question is about to be answered
        self.lang.dialogue.unanswered_Q.remove(ui)

        dialogue_state = self.lang.dialogue.export_as_dict()
        _, _, _, orig_utt = dialogue_state["record"][ui]

        translated = self.recognitive.translate_dialogue_content(dialogue_state)

        _, query = translated[ui]
        assert query is not None

        q_vars, _ = query

        # Compute raw answer candidates by appropriately querying current world models
        models_vl, _, _ = self.recognitive.concl_vis_lang
        answers_raw, _ = models_vl.query(*query)

        if q_vars is not None:
            # (Temporary) For now, let's limit our answer to "what is X" questions to nouns:
            # i.e. object class categories...
            answers_raw = {
                ans: val for ans, val in answers_raw.items()
                if any(not is_pred or a.startswith("cls") for a, (_, is_pred) in zip(ans, q_vars))
            }

        # Pick out an answer to deliver; maximum confidence
        if len(answers_raw) > 0:
            answer_selected = max(answers_raw, key=lambda a: answers_raw[a][1])
            _, ev_prob = answers_raw[answer_selected]
        else:
            answer_selected = (None,) * len(q_vars)
            ev_prob = None

        # Translate the selected answer into natural language
        # (Parse the original question utterance, manipulate, then generate back)
        if len(answer_selected) == 0:
            # Yes/no question; cast original question to proposition
            answer_translated = self.lang.semantic.nl_change_sf(orig_utt, "prop")

            if ev_prob < 0.5:
                # Flip polarity for negative answer with event probability lower than 0.5
                answer_translated = self.lang.semantic.nl_negate(answer_translated)
        else:
            # Wh- question; replace wh-quantified referent(s) with appropriate answer values
            answer_translated = orig_utt

            replace_targets = []
            replace_values = []

            for (qv, is_pred), ans in zip(q_vars, answer_selected):
                # Char range in original utterance, referring to expression to be replaced
                tgt = dialogue_state["referents"]["dis"][qv]["provenance"]
                replace_targets.append(tgt)

                low_confidence = ev_prob is not None and ev_prob < 0.5

                # Value to replace the designated wh-quantified referent with
                if is_pred:
                    # Predicate name; fetch from lexicon
                    if ans is None or low_confidence:
                        # No answer predicate to "What is X" question; let's simply generate
                        # "I am not sure" as answer for these cases
                        self.lang.dialogue.to_generate.append("I am not sure.")
                        return
                    else:
                        ans = ans.split("_")
                        ans = (int(ans[1]), ans[0])

                        is_named = False
                        val = self.lt_mem.lexicon.d2s[ans][0][0]
                else:
                    # Entity by their constant name handle
                    is_named = True
                    if low_confidence:
                        val = None
                    else:
                        val = ans

                replace_values.append((val, is_named))

            # Plug in the selected answer in place of the wh-quantified referent
            answer_translated = self.lang.semantic.nl_replace_wh(
                answer_translated, replace_targets, replace_values
            )

            # Don't forget to make it a prop
            answer_translated = self.lang.semantic.nl_change_sf(answer_translated, "prop")

        # Push the translated answer to buffer of utterances to generate
        self.lang.dialogue.to_generate.append(answer_translated)
