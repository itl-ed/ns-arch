"""
Outermost wrapper containing ITL agent API
"""
import os
import math
import pickle
import readline
import rlcompleter

import torch
from detectron2.structures import BoxMode

from .memory import LongTermMemoryModule
from .vision import VisionModule
from .vision.utils.completer import DatasetImgsCompleter
from .lang import LanguageModule
from .theoretical_reasoning import TheoreticalReasonerModule
from .practical_reasoning import PracticalReasonerModule
from .actions import AgentCompositeActions
from .lpmln import Rule, Literal
from .lpmln.utils import wrap_args
from .path_manager import PathManager


# FT_THRES = 0.5              # Few-shot learning trigger score threshold
SR_THRES = -math.log(0.5)    # Mismatch surprisal threshold
U_IN_PR = 1.0                # How much the agent values information provided by the user
EPS = 1e-10                  # Value used for numerical stabilization
TAB = "\t"                   # For use in format strings

class ITLAgent:

    def __init__(self, opts):
        self.opts = opts
        self.path_manager = PathManager

        # Initialize component modules
        self.vision = VisionModule(opts)
        self.lang = LanguageModule(opts)
        self.theoretical = TheoreticalReasonerModule()
        self.practical = PracticalReasonerModule()
        self.lt_mem = LongTermMemoryModule()

        # Register 'composite' agent actions that require more than one modules to
        # interact
        self.comp_actions = AgentCompositeActions(self)

        # Load agent model from specified path
        self.load_model()

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

    def test_binary(self, v_input, target_bbox, concepts):
        """
        Surgically (programmatically) test the agent's performance with an exam question,
        without having to communicate through full 'natural interactions...
        """
        # Run only the backbone to cache image feature maps
        inp = { "file_name": v_input }
        inp = [self.vision.dm.mapper_batch["test"](inp)]
        images = self.vision.model.base_model.preprocess_image(inp)
        features = self.vision.model.base_model.backbone(images.tensor)

        # Vision module buffer cleanup
        self.vision.scene = {}
        self.vision.f_vecs = ({}, {}, {}, {}, {}, features)

        # Make prediction on the designated bbox
        bboxes = {
            "o0": {
                "bbox": target_bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "objectness_scores": None
            }
        }
        self.vision.predict(
            v_input, exemplars=self.lt_mem.exemplars, bboxes=bboxes
        )

        # Inform the language module of the visual context
        self.lang.situate(self.vision.last_input, self.vision.scene)
        self.theoretical.refresh()

        # Sensemaking from vision input only
        exported_kb = self.lt_mem.kb.export_reasoning_program(self.vision.scene)
        self.theoretical.sensemake_vis(self.vision.scene, exported_kb)
        models_v, _, _ = self.theoretical.concl_vis

        # "Question" for probing agent's performance
        q_vars = (("P", True),)
        q_rules = [Rule(head=Literal("*_?", wrap_args("P", "o0")))]
        question = (q_vars, q_rules)

        # Parts taken from self.comp_actions.prepare_answer_Q(), excluding processing
        # questions in dialogue records
        if len(self.lt_mem.kb.entries) > 0:
            search_specs = self.comp_actions._search_specs_from_kb(question, models_v)
            if len(search_specs) > 0:
                self.vision.predict(
                    self.vision.last_input, exemplars=self.lt_mem.exemplars,
                    specs=search_specs
                )

            exported_kb = self.lt_mem.kb.export_reasoning_program(self.vision.scene)
            self.theoretical.sensemake_vis(self.vision.scene, exported_kb)
        
        answers_raw, _ = models_v.query(*question)

        # Collate results with respect to the provided concept set
        denotations = [
            "".join(
                tok.capitalize() if i>0 else tok
                for i, tok in enumerate(c.split(".")[0].split("_"))
            )
            for c in concepts
        ]
        denotations = [self.lt_mem.lexicon.s2d[(c, "n")][0] for c in denotations]
        denotations = [f"{cat_type}_{conc_ind}" for conc_ind, cat_type in denotations]

        agent_answers = {
            c: True if (d,) in answers_raw and answers_raw[(d,)][1] > 0.5 else False
            for c, d in zip(concepts, denotations)
        }

        return agent_answers

    def save_model(self, ckpt_path):
        """
        Save current snapshot of the agent's long-term knowledge as torch checkpoint;
        in the current scope of the research, by 'long-term knowledge' we are referring
        to the following information:

        - Vision model; feature extractor backbone most importantly, and concept-specific
            vectors
        - Knowledge stored in long-term memory module:
            - Symbolic knowledge base, including generalized symbolic knowledge represented
                as logic programming rules
            - Visual exemplar base, including positive/negative exemplars of visual concepts
                represented as internal feature vectors along with original image patches from
                which the vectors are obtained
            - Lexicon, including associations between words (linguistic form) and their
                denotations (linguistic meaning; here, visual concepts)
        """
        ckpt = {
            "vision": {
                "state_dict": self.vision.model.state_dict(),
                "predicates": self.vision.predicates,
                "predicates_freq": self.vision.predicates_freq
            },
            "lt_mem": {
                "exemplars": vars(self.lt_mem.exemplars),
                "kb": vars(self.lt_mem.kb),
                "lexicon": vars(self.lt_mem.lexicon)
            }
        }
        torch.save(ckpt, ckpt_path)

    def load_model(self):
        """
        Load from a torch checkpoint to initialize the agent; the checkpoint may contain
        a snapshot of agent knowledge obtained as an output of self.save_model() evoked
        previously, or just pre-trained weights of the vision module only (likely generated
        as output of the vision module's training API)
        """
        # Resolve path to checkpoint (Use vision cfg default if not provided)
        ckpt_path = self.opts.load_checkpoint_path or self.vision.cfg.MODEL.WEIGHTS
        if not os.path.isfile(ckpt_path):
            local_ckpt_path = self.path_manager.get_local_path(ckpt_path)
            assert os.path.isfile(local_ckpt_path), \
                "Checkpoint {} not found!".format(local_ckpt_path)
        else:
            local_ckpt_path = ckpt_path

        # Load agent model checkpoint file
        try:
            ckpt = torch.load(local_ckpt_path)
        except RuntimeError:
            with open(local_ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

        if "vision" in ckpt:
            # Likely a checkpoint saved with self.save_model(); first load the vision
            # module
            self.vision.load_model(ckpt["vision"])

            # Fill in long-term memory components with loaded data
            for mem_component, mem_data in ckpt["lt_mem"].items():
                for component_prop, component_data in mem_data.items():
                    component = getattr(self.lt_mem, mem_component)
                    setattr(component, component_prop, component_data)
        else:
            # Likely a checkpoint only containing pre-trained vision model weights;
            # pass the dict directly to self.vision.load_model()
            self.vision.load_model(ckpt)

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
                elif usr_in == "exit":
                    print(f"Sys> Terminating...")
                    quit()
                else:
                    if usr_in not in self.dcompleter:
                        raise ValueError(f"Image file {usr_in} does not exist")
                    self.vision.new_input = usr_in

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

        # Index of latest utterance
        ui_last = len(self.lang.dialogue.record)

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
                    self.vision.last_input, exemplars=self.lt_mem.exemplars,
                    visualize=vis_ui_on
                )
                vis_ui_on = False

                # Temporary tampering for the sake of efficient experiments; nuke ensemble
                # prediction result...
                self.vision.scene = {}
                self.vision.f_vecs = ({}, {}, {}, {}, {}, self.vision.f_vecs[5])

            if self.vision.new_input is not None:
                # Inform the language module of the visual context
                self.lang.situate(self.vision.last_input, self.vision.scene)
                self.theoretical.refresh()

                # No need to treat these facts as 'mismatches'
                # (In a sense, this kinda overlaps with the notion of 'common ground'?
                # May consider fulfilling this later)
                self.doubt_no_more = set()

            # Understand the user input in the context of the dialogue
            if self.lang.new_input is not None and self.lang.new_input["raw"] != "Correct.":
                self.lang.dialogue.record = self.lang.dialogue.record[:ui_last]
                self.lang.understand(
                    self.lang.new_input, self.vision.last_input, pointing=pointing
                )

            if self.vision.scene is not None:
                # If a new entity is registered as a result of understanding the latest
                # input, re-run vision module to update with new predictions for it
                new_ents = set(self.lang.dialogue.referents["env"]) - set(self.vision.scene)
                if len(new_ents) > 0:
                    bboxes = {
                        ent: {
                            "bbox": self.lang.dialogue.referents["env"][ent]["bbox"],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "objectness_scores": None
                        }
                        for ent in new_ents
                    }

                    # Incrementally predict on the designated bbox
                    self.vision.predict(
                        self.vision.last_input, exemplars=self.lt_mem.exemplars,
                        bboxes=bboxes
                    )

            ###################################################################
            ##       Sensemaking via synthesis of perception+knowledge       ##
            ###################################################################

            dialogue_state = self.lang.dialogue.export_as_dict()

            if self.vision.new_input is not None or vision_model_updated or kb_updated:
                # Sensemaking from vision input only
                exported_kb = self.lt_mem.kb.export_reasoning_program(self.vision.scene)
                self.theoretical.sensemake_vis(self.vision.scene, exported_kb)

            if self.lang.new_input is not None:
                # Reference & word sense resolution to connect vision & discourse
                self.theoretical.resolve_symbol_semantics(
                    dialogue_state, self.lt_mem.lexicon
                )

                if self.vision.scene is not None:
                    # Sensemaking from vision & language input
                    self.theoretical.sensemake_vis_lang(dialogue_state)

            ###################################################################
            ##           Identify & exploit learning opportunities           ##
            ###################################################################

            # Resetting flags
            vision_model_updated = False
            kb_updated = False

            # Process translated dialogue record to do the following:
            #   - Integrate newly provided generic rules into KB
            #   - Identify recognition mismatch btw. user provided vs. agent
            translated = self.theoretical.translate_dialogue_content(dialogue_state)
            for ui, (rules, _) in enumerate(translated):
                kb_rules_to_add = []
                if rules is not None:
                    for r in rules:
                        # Symbolic knowledge base expansion; for generic rules without
                        # constant terms
                        if all(is_var for _, is_var in r.terms()):
                            # Integrate the rule into KB by adding (for now we won't
                            # worry about intra-KB consistency, etc.)
                            kb_rules_to_add.append(r)

                        # Test against vision-only sensemaking result to identify any
                        # theoretical mismatch
                        if all(not is_var for _, is_var in r.terms()) \
                            and self.theoretical.concl_vis is not None:
                            if r in self.doubt_no_more:
                                # May skip testing this one
                                continue

                            # Make a yes/no query to obtain the likelihood of content
                            models_v, _, _ = self.theoretical.concl_vis
                            q_response, _ = models_v.query(None, r)
                            ev_prob = q_response[()][1]

                            surprisal = -math.log(ev_prob + EPS)
                            if surprisal > SR_THRES:
                                m = (r, surprisal)
                                self.theoretical.mismatches.add(m)
                    
                    if len(kb_rules_to_add) > 0:
                        provenance = dialogue_state["record"][ui][2]
                        kb_updated |= self.lt_mem.kb.add(kb_rules_to_add, U_IN_PR, provenance)

            # Handle neologisms
            neologisms = {
                tok: sym for tok, (sym, den) in self.theoretical.word_senses.items()
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
                    novel_concept = (concept_ind, cat_type)

                    # Acquire novel concept by updating lexicon (and vision.predicates)
                    self.lt_mem.lexicon.add((name, pos), novel_concept)
                    # self.vision.predicates[cat_type].append(
                    #     f"{name}.{pos}.0{len(self.lt_mem.lexicon.s2d[(name, pos)])}"
                    # )

                    ui = int(tok[0].strip("u"))
                    ri = int(tok[1].strip("r"))
                    rule_head, rule_body, _ = dialogue_state["record"][ui][1][0][ri]

                    if rule_body is None:
                        # Labelled exemplar provided; add new concept exemplars to
                        # memory, as feature vectors at the penultimate layer right
                        # before category prediction heads
                        args = [
                            self.theoretical.value_assignment[arg] for arg in rule_head[0][2]
                        ]
                        ex_bboxes = [
                            self.lang.dialogue.referents["env"][a]["bbox"] for a in args
                        ]

                        if cat_type == "cls":
                            f_vec = self.vision.f_vecs[0][args[0]]
                        elif cat_type == "att":
                            f_vec = self.vision.f_vecs[1][args[0]]
                        else:
                            assert cat_type == "rel"
                            f_vec = self.vision.f_vecs[2][args[0]][args[1]]
                        
                        pointers_src = { 0: (0, tuple(ai for ai in range(len(args)))) }
                        pointers_exm = { concept_ind: ({0}, set()) }

                        self.lt_mem.exemplars.add_exs(
                            sources=[(self.vision.last_raw, ex_bboxes)],
                            f_vecs={ cat_type: f_vec[None,:].cpu().numpy() },
                            pointers_src={ cat_type: pointers_src },
                            pointers_exm={ cat_type: pointers_exm }
                        )

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

        # Ideally, this is to be accomplished declaratively by properly setting up formal
        # maintenance goals and then performing automated planning or something to come
        # up with right sequence of actions to be added to agenda. However, the ad-hoc code
        # below (+ plan library in practical/plans/library.py) will do for our purpose right
        # now; we will see later if we'll ever need to generalize and implement the said
        # procedure.)

        for n in self.lang.unresolved_neologisms:
            self.practical.agenda.append(("address_neologism", n))
        for m in self.theoretical.mismatches:
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
