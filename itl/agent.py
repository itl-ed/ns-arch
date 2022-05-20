"""
Outermost wrapper containing ITL agent API
"""
import math
import readline
import rlcompleter

from detectron2.structures import BoxMode

from .memory import LongTermMemoryModule
from .vision import VisionModule
from .vision.utils.completer import DatasetImgsCompleter
from .lang import LanguageModule
from .cognitive_reasoning import CognitiveReasonerModule
from .practical_reasoning import PracticalReasonerModule
from .lpmln import Literal, Rule
from .lpmln.utils import wrap_args


FT_THRES = 0.5                # Few-shot learning trigger score threshold
SR_THRES = -math.log(0.5)     # Mismatch surprisal threshold
EPS = 1e-10                   # Value used for numerical stabilization
TAB = "\t"                    # For use in format strings

class ITLAgent:

    def __init__(self, opts):
        # Initialize component modules
        self.lt_mem = LongTermMemoryModule()
        self.vision = VisionModule(opts)
        self.lang = LanguageModule(opts)
        self.cognitive = CognitiveReasonerModule(kb=self.lt_mem.kb)
        self.practical = PracticalReasonerModule()

        # Initialize empty lexicon with concepts in visual module
        self.lt_mem.lexicon.fill_from_vision(self.vision)

        # Image file selection CUI
        self.dcompleter = DatasetImgsCompleter()
        readline.parse_and_bind("tab: complete")

    def __call__(self):
        """Main function: Kickstart infinite ITL agent loop"""
        print(f"Sys> At any point, enter 'exit' to quit")

        while True:
            self._vis_inp()
            self._lang_inp()
            self._update_belief()
            self._act()
    
    def _vis_inp(self):
        """Image input prompt (Choosing from dataset for now)"""
        # Register autocompleter for REPL
        readline.set_completer(self.dcompleter.complete)

        print("")

        while True:

            print(f"Sys> Choose an image to process")
            print("Sys> Enter 'r' for random selection, 'n' for skipping new image input")
            usr_in = input("U> ")
            print("")

            try:
                if usr_in == "n":
                    print(f"Sys> Skipped image selection")
                    img_f = None
                    break
                elif usr_in == "r":
                    img_f = self.dcompleter.sample()
                elif usr_in == "exit":
                    print(f"Sys> Terminating...")
                    quit()
                else:
                    if usr_in not in self.dcompleter:
                        raise ValueError(f"Image file {usr_in} does not exist")
                    img_f = usr_in

            except ValueError as e:
                print(f"Sys> {e}, try again")

            else:
                print(f"Sys> Selected image file: {img_f}")
                break

        if img_f is not None:
            # Run visual inference on designated image, and store raw image & generated
            # scene graph
            self.vision.predict(img_f, visualize=True)
        
        # Restore default completer
        readline.set_completer(rlcompleter.Completer().complete)
    
    def _lang_inp(self):
        """Language input prompt (from user)"""
        if self.vision.updated:
            # Inform the language module of the visual context
            self.lang.situate(self.vision.raw_input, self.vision.scene)
            self.cognitive.refresh()

        print("")
        print("Sys> Awaiting user input...")
        print("Sys> Enter 'n' for skipping language input")

        valid_input = False
        while not valid_input:
            usr_in = input("U> ")
            print("")

            # Understand the user input in the context of the dialogue
            try:
                if usr_in == "n":
                    print(f"Sys> Skipped language input")
                    break
                elif usr_in == "exit":
                    print(f"Sys> Terminating...")
                    quit()
                else:
                    self.lang.understand(usr_in, self.vision.raw_input)

            except IndexError as e:
                print(f"Sys> Ungrammatical input or IndexError: {e.args}")
                continue
            except ValueError as e:
                print(f"Sys> {e.args[0]}")
                continue
            except NotImplementedError:
                print("Sys> Sorry, can't handle the input sentence (yet)")
                continue

            else:
                valid_input = True
                break

        if self.vision.scene is not None:
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
                raw_input_bgr = self.vision.raw_input[:, :, [2,1,0]]
                self.vision.predict(raw_input_bgr, bboxes=bboxes)
    
    def _update_belief(self):
        """ Form beliefs based on visual and/or language input """

        if self.vision.raw_input is None and len(self.lang.dialogue.record) == 0:
            # No information whatsoever to form any sort of beliefs
            print("A> (Idling the moment away...)")
            return

        # Handle resolvable/unresolvable neologisms, if any
        self._handle_neologisms()

        # Sensemaking from vision input only
        if self.vision.updated:
            self.cognitive.sensemake_vis(self.vision.scene)

        dialogue_state = self.lang.export_dialogue_state()

        # Reference & word sense resolution to connect vision & discourse
        self.cognitive.resolve_symbol_semantics(dialogue_state, self.lt_mem.lexicon)

        # Learning from user language input at two levels: neural few-shot learning
        # and symbolic rule learning
        self._learn()

        # Sensemaking from vision & language input
        self.cognitive.sensemake_vis_lang(dialogue_state)

        # Identify any mismatch between vision-only sensemaking result vs. info conveyed
        # by user utterance inputs
        self._identify_mismatch()

        # Compute answers to any unanswered questions that can be computed
        self._compute_Q_answers()

        # self.vision.reshow_pred()

    def _act(self):
        """
        Just eagerly try to resolve each item in agenda as much as possible, generating
        and performing actions until no more agenda items can be resolved for now. I
        wonder if we'll ever need a more sophisticated mechanism than this simple, greedy
        method for a good while?
        """
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

                            act_method(*act_args)

                    resolved_items.append(i)

            if len(resolved_items) == 0:
                # No resolvable agenda item any more; break
                break
            else:
                # Check off resolved agenda item
                resolved_items.reverse()
                for i in resolved_items:
                    del self.practical.agenda[i]

    ##################################################################################
    ##  Below implements agent capabilities that require interplay between modules  ##
    ##################################################################################

    def _handle_neologisms(self):
        """
        Identify any neologisms that can(not) be resolved without interacting further
        with user (definition/exemplar already provided)
        """
        neologisms = set()
        resolvable_neologisms = set()
        for _, _, (rules, query), _ in self.lang.dialogue.record:
            if rules is not None:
                for head, body, _ in rules:
                    if head is not None:
                        if head[:2] not in self.lt_mem.lexicon.s2d:
                            # Occurrence in rule head implies either definition or exemplar
                            # is provided by this utterance
                            resolvable_neologisms.add(head[:2])
                            neologisms.add(head[:2])
                    if body is not None:
                        for b in body:
                            if b[:2] not in self.lt_mem.lexicon.s2d: neologisms.add(b[:2])
            if query is not None:
                _, q_fmls = query
                for head, body, _ in q_fmls:
                    if head is not None:
                        if head[:2] not in self.lt_mem.lexicon.s2d: neologisms.add(head[:2])
                    if body is not None:
                        for b in body:
                            if b[:2] not in self.lt_mem.lexicon.s2d: neologisms.add(b[:2])

        unresolvable_neologisms = neologisms - resolvable_neologisms

        if len(resolvable_neologisms) > 0:
            # Assign each novel concept a new index, while initializing the class code for
            # the index as a NaN vector
            for n in resolvable_neologisms:
                symbol, pos = n
                if pos == "n":
                    cat_type = "cls"
                elif pos == "a":
                    cat_type = "att"
                else:
                    assert pos == "v" or pos == "r"
                    cat_type = "rel"

                # Expand corresponding visual concept inventory
                concept_ind = self.vision.add_concept(cat_type)

                # Update lexicon (and vision.predicates)
                self.lt_mem.lexicon.add(n, (concept_ind, cat_type))
                self.vision.predicates[cat_type].append(
                    f"{symbol}.{pos}.0{len(self.lt_mem.lexicon.s2d[n])}"
                )

        if len(unresolvable_neologisms) > 0:
            # Add as agenda item to request what each unresolvable neologism 'means'
            for n in unresolvable_neologisms:
                self.practical.agenda.append(("address_neologism", n))

    def _learn(self):
        """
        All the actual 'learning' from user interactions happens here (after all, learning
        is a form of belief update). An instance of lifelong learning?
        """
        # Shortcut vars
        a_map = lambda args: [self.cognitive.value_assignment[a] for a in args]
        word_senses = self.cognitive.word_senses

        for speaker, _, (rules, _), _ in self.lang.dialogue.record:
            if speaker == "U" and rules is not None:
                for head, body, _ in rules:
                    # Neural (low-level) learning: Happens when 'reflex'-perception from neural
                    # sensor module doesn't agree with provided info
                    if body is None:
                        vis_concept = word_senses[head[:2]]
                        cat_type, cat_ind = vis_concept

                        args = a_map(head[2])

                        # Fetch current score for the asserted fact (if exists)
                        if cat_type == "cls":
                            cat_scores = self.vision.scene[args[0]]["pred_classes"]
                            f_vec = self.vision.f_vecs[0][args[0]]
                        elif cat_type == "att":
                            cat_scores = self.vision.scene[args[0]]["pred_attributes"]
                            f_vec = self.vision.f_vecs[1][args[0]]
                        else:
                            assert cat_type == "rel"
                            cat_scores = self.vision.scene[args[0]]["pred_relations"][args[1]]
                            f_vec = self.vision.f_vecs[2][args[0]][args[1]]

                        # If score doesn't exist (due to being novel concept) or score is below the
                        # predefined threshold, trigger few-shot learning
                        if cat_ind >= len(cat_scores) or cat_scores[cat_ind] < FT_THRES:
                            # Add new concept exemplars to memory, as feature vectors at the penultimate
                            # layer right before category prediction heads
                            self.lt_mem.exemplars.add(vis_concept, f_vec.cpu().numpy())

                            # Update the category code parameter in the vision model's predictor head
                            # using the new set of exemplars
                            self.vision.update_concept(vis_concept, self.lt_mem.exemplars[vis_concept])

                            # Re-run vision prediction with updated model then sensemake again
                            raw_input_bgr = self.vision.raw_input[:, :, [2,1,0]]
                            self.vision.predict(raw_input_bgr, bboxes=self.vision.latest_bboxes)
                            self.cognitive.sensemake_vis(self.vision.scene)

                    # Symbolic (high-level) learning: Add any novel generic rules to knowledge
                    # base; rule shouldn't contain any constant term to be considered generic
                    print(0)

            # occurring_lits = sum([
            #     ([r[0]] if r[0] is not None else []) + (r[1] if r[1] is not None else [])
            #     for r in info+info_aux
            # ], [])
            # occurring_args = set(sum([
            #     sum([a[1] if type(a)==tuple else (a,) for a in l[2]], ()) for l in occurring_lits
            # ], ()))

            # If rule contains unresolved neologism postpone knowledge integration
            # unresolved_neologisms = {
            #     ag_content for (ag_type, ag_content) in self.practical.agenda
            #     if ag_type == "address_neologism"
            # }
            # if rules is not None:
            #     _, q_fmls = query

            #     skip_query = False
            #     for head, body, _ in q_fmls:
            #         if head is not None:
            #             if head[:2] in unresolved_neologisms:
            #                 skip_query = True; break
            #         if body is not None:
            #             for b in body:
            #                 if b[:2] in unresolved_neologisms:
            #                     skip_query = True; break
            #     if skip_query:
            #         continue

            # if all(a[0].isupper() for a in occurring_args):
            #     agenda.append(("unintegrated_knowledge", ui))

    def _identify_mismatch(self):
        """
        Recognize any mismatch between sensemaking results obtained from vision-only
        vs. info provided in discourse utterances, and add to agenda if any is found
        """
        if len(self.lang.dialogue.record) == 0:
            # Don't bother
            return

        # Shortcut vars
        a_map = lambda args: [self.cognitive.value_assignment[a] for a in args]
        word_senses = self.cognitive.word_senses
        models_v, _, _ = self.cognitive.concl_vis

        for _, _, (rules, _), _ in self.lang.dialogue.record:
            if rules is not None:
                content = set()
                for rule in rules:
                    head, body, _ = rule

                    # Skip any non-grounded content
                    head_has_var = head is not None and any([
                        type(x)==str and x[0].isupper() for x in head[2]
                    ])
                    body_has_var = body is not None and any([
                        any([type(x)==str and x[0].isupper() for x in bl[2]])
                        for bl in body
                    ])
                    if head_has_var or body_has_var: continue

                    if head is not None:
                        pred = "_".join([str(s) for s in word_senses[head[:2]]])
                        args = [a for a in a_map(head[2])]
                        subs_head = Literal(pred, wrap_args(*args))
                    else:
                        subs_head = None
                    
                    if body is not None:
                        subs_body = []
                        for bl in body:
                            pred = "_".join([str(s) for s in word_senses[bl[:2]]])
                            args = [a for a in a_map(bl[2])]
                            bl = Literal(pred, wrap_args(*args), naf=bl[3])
                            subs_body.append(bl)
                    else:
                        subs_body = None
                    
                    content.add(Rule(head=subs_head, body=subs_body))

                # Make a yes-no query to obtain the likelihood of content
                q_response, _ = models_v.query(None, content)
                ev_prob = q_response[()][1]

                surprisal = -math.log(ev_prob + EPS)
                if surprisal > SR_THRES:
                    m = (content, surprisal)
                    self.practical.agenda.append(("address_mismatch", m))
        
        ## TODO: Print surprisal reports to resolve_mismatch action ##
        # print("A> However, I was quite surprised to hear that:")

        # for m in mismatches:
        #     is_positive = m[0]     # 'True' stands for positive statement

        #     if is_positive:
        #         # Positive statement (fact)
        #         message = f"{m[2][2][0]} is (a/an) {m[1][0]}"
        #     else:
        #         # Negative statement (constraint)
        #         negated = [
        #             f"{m2[2][0]} is (a/an) {m1[0]}" for m1, m2 in zip(m[1], m[2])
        #         ]
        #         message = f"Not {{{' & '.join(negated)}}}"

        #     print(f"A> {TAB}{message} (surprisal: {round(m[3], 3)})")

    def _compute_Q_answers(self):
        """ Compute answers to question indexed by indexed by utt_id """
        if len(self.lang.dialogue.record) == 0:
            # Don't bother
            return

        assert self.cognitive.concl_vis_lang is not None
        models_vl, _, _ = self.cognitive.concl_vis_lang

        for utt_id, (_, utt_type, (_, query), _) in enumerate(self.lang.dialogue.record):
            if utt_type == "?" and utt_id not in self.cognitive.Q_answers:
                # If query contains unresolved neologism postpone answer computation
                unresolved_neologisms = {
                    ag_content for (ag_type, ag_content) in self.practical.agenda
                    if ag_type == "address_neologism"
                }
                if query is not None:
                    _, q_fmls = query

                    skip_query = False
                    for head, body, _ in q_fmls:
                        if head is not None:
                            if head[:2] in unresolved_neologisms:
                                skip_query = True; break
                        if body is not None:
                            for b in body:
                                if b[:2] in unresolved_neologisms:
                                    skip_query = True; break
                    if skip_query:
                        continue

                    # Compute answer to the query
                    _, query = self.lang.utt_to_ASP(utt_id)
                    q_ents, q_rules, _ = query

                    # Store the computed answer, and add to agenda to generate answer
                    # utterance
                    self.cognitive.Q_answers[utt_id] = models_vl.query(q_ents, q_rules)
                    self.practical.agenda.append(("answer_Q", utt_id))
