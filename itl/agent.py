"""
Outermost wrapper containing ITL agent API
"""
import os
import copy
import math
import pickle
import logging
from collections import defaultdict
# import readline
# import rlcompleter

import torch
import numpy as np
from torchvision.ops import box_convert
from pytorch_lightning.loggers import WandbLogger

from .memory import LongTermMemoryModule
from .vision import VisionModule
from .lang import LanguageModule
from .symbolic_reasoning import SymbolicReasonerModule
from .practical_reasoning import PracticalReasonerModule
from .actions import AgentCompositeActions
from .lpmln import Rule, Literal
from .lpmln.utils import wrap_args
# from .utils.completer import DatasetImgsCompleter

logger = logging.getLogger(__name__)


SR_THRES = 0.5               # Mismatch surprisal threshold
SC_THRES = 0.3               # Binary decision score threshold
U_IN_PR = 1.00               # How much the agent values information provided by the user
A_IM_PR = 0.80               # How much the agent values inferred implicature
EPS = 1e-10                  # Value used for numerical stabilization
TAB = "\t"                   # For use in format strings

WB_PREFIX = "wandb://"

class ITLAgent:

    def __init__(self, cfg):
        self.cfg = cfg

        # Initialize component modules
        self.vision = VisionModule(cfg)
        self.lang = LanguageModule(cfg)
        self.symbolic = SymbolicReasonerModule()
        self.practical = PracticalReasonerModule()
        self.lt_mem = LongTermMemoryModule()

        # Register 'composite' agent actions that require more than one modules to
        # interact
        self.comp_actions = AgentCompositeActions(self)

        # Load agent model from specified path
        if "model_path" in cfg.agent:
            self.load_model(cfg.agent.model_path)

        # Agent learning strategy params
        self.strat_generic = cfg.agent.strat_generic

        # Show visual UI and plots
        self.vis_ui_on = True

        # Image file selection CUI
        # self.dcompleter = DatasetImgsCompleter(cfg.paths.data_dir)
        # readline.parse_and_bind("tab: complete")

        # (Fields below would categorize as 'working memory' in conventional
        # cognitive architectures...)

        # Bookkeeping pairs of visual concepts that confused the agent, which
        # are resolved by asking 'concept-diff' questions to the user. Jusk ask
        # once to get answers as symbolic generic rules when the agent is aware
        # of the confusion for the first time, for each concept pair.
        # (In a sense, this kinda overlaps with the notion of 'common ground'?
        # May consider fulfilling this later...)
        self.confused_no_more = set()

        # Snapshot of KB, to be taken at the beginning of every training episode,
        # with which scalar implicatures will be computed. Won't be necessary
        # if we were to use more structured discourse representation...
        self.kb_snap = copy.deepcopy(self.lt_mem.kb.entries_by_pred)

        # (Temporary) Restrict our attention to newly acquired visual concepts
        # when it comes to dealing with concept confusions
        self.cls_offset = self.vision.inventories.cls

    def __call__(self):
        """Main function: Kickstart infinite ITL agent loop with user interface"""
        logger.info("Sys> At any point, enter 'exit' to quit")

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
        # Vision module buffer cleanup
        self.vision.scene = {}
        self.vision.f_vecs = {}

        # First, ensemble prediction
        self.vision.predict(
            v_input, self.lt_mem.exemplars,
            visualize=False, lexicon=self.lt_mem.lexicon
        )

        # Make prediction on the designated bbox if needed
        bboxes = {
            "o0": {
                "bbox": target_bbox,
                "bbox_mode": "xyxy"
            }
        }
        self.vision.predict(
            None, self.lt_mem.exemplars, bboxes=bboxes, visualize=False
        )

        # Inform the language module of the visual context
        self.lang.situate(self.vision.last_input, self.vision.scene)
        self.symbolic.refresh()

        # Sensemaking from vision input only
        exported_kb = self.lt_mem.kb.export_reasoning_program()
        self.symbolic.sensemake_vis(self.vision.scene, exported_kb)
        bjt_v, _ = self.symbolic.concl_vis

        # "Question" for probing agent's performance
        question = (
            (("P", True),),
            ((Literal("*_?", wrap_args("P", "o0")),), None)
        )

        # Parts taken from self.comp_actions.prepare_answer_Q(), excluding processing
        # questions in dialogue records
        if len(self.lt_mem.kb.entries) > 0:
            search_specs = self.comp_actions._search_specs_from_kb(question, bjt_v)
            if len(search_specs) > 0:
                self.vision.predict(
                    None, self.lt_mem.exemplars, specs=search_specs, visualize=False
                )

            exported_kb = self.lt_mem.kb.export_reasoning_program()
            self.symbolic.sensemake_vis(self.vision.scene, exported_kb)
            bjt_v, _ = self.symbolic.concl_vis
        
        answers_raw = self.symbolic.query(bjt_v, *question)

        # Collate results with respect to the provided concept set
        denotations = [
            "".join(
                tok.capitalize() if i>0 else tok
                for i, tok in enumerate(c.split(".")[0].split("_"))
            )
            for c in concepts
        ]
        denotations = [self.lt_mem.lexicon.s2d[(c, "n")][0] for c in denotations]
        denotations = [f"{conc_type}_{conc_ind}" for conc_ind, conc_type in denotations]

        agent_answers = {
            c: (d,) in answers_raw and answers_raw[(d,)] > SC_THRES
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
                "inventories": self.vision.inventories,
                "fs_model_path": self.cfg.vision.model.fs_model
            },
            "lt_mem": {
                "exemplars": vars(self.lt_mem.exemplars),
                "kb": vars(self.lt_mem.kb),
                "lexicon": vars(self.lt_mem.lexicon)
            }
        }
        torch.save(ckpt, ckpt_path)
        logger.info(f"Saved current agent model at {ckpt_path}")

    def load_model(self, ckpt_path):
        """
        Load from a torch checkpoint to initialize the agent; the checkpoint may contain
        a snapshot of agent knowledge obtained as an output of self.save_model() evoked
        previously, or just pre-trained weights of the vision module only (likely generated
        as output of the vision module's training API)
        """
        # Resolve path to checkpoint
        if ckpt_path.startswith(WB_PREFIX):
            # Storing agent models in W&B; not implemented yet
            raise NotImplementedError

            # wb_entity = os.environ.get("WANDB_ENTITY")
            # wb_project = os.environ.get("WANDB_PROJECT")
            # wb_run_id = self.fs_model_path[len(WB_PREFIX):]

            # local_ckpt_path = WandbLogger.download_artifact(
            #     artifact=f"{wb_entity}/{wb_project}/model-{wb_run_id}:best_k",
            #     save_dir=os.path.join(
            #         self.cfg.paths.assets_dir, "vision_models", "wandb", wb_run_id
            #     )
            # )
            # local_ckpt_path = os.path.join(local_ckpt_path, "model.ckpt")
        else:
            assert os.path.exists(ckpt_path)
            local_ckpt_path = ckpt_path

        # Load agent model checkpoint file
        try:
            ckpt = torch.load(local_ckpt_path)
        except RuntimeError:
            with open(local_ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

        # Fill in module components with loaded data
        for module_name, module_data in ckpt.items():
            for module_component, component_data in module_data.items():
                if isinstance(component_data, dict):
                    for component_prop, prop_data in component_data.items():
                        component = getattr(getattr(self, module_name), module_component)
                        setattr(component, component_prop, prop_data)
                else:
                    module = getattr(self, module_name)
                    prev_component_data = getattr(module, module_component)
                    setattr(module, module_component, component_data)
                
                # Handling vision.fs_model_path data
                if module_name == "vision":
                    if module_component == "fs_model_path":
                        if (prev_component_data is not None and
                            prev_component_data != component_data):
                            logger.warn(
                                "Path to few-shot components in vision module is already provided "
                                "in config and is inconsistent with the pointer saved in the agent "
                                f"model specified (config: {prev_component_data} vs. agent_model: "
                                f"{component_data}). Agent vision module might exhibit unexpected "
                                "behaviors."
                            )
                        
                        self.vision.load_weights()

    def _vis_inp(self, usr_in=None):
        """Image input prompt (Choosing from dataset for now)"""
        self.vision.new_input = None
        input_provided = usr_in is not None

        # Register autocompleter for REPL
        # readline.set_completer(self.dcompleter.complete)

        while True:

            logger.debug("Sys> Choose an image to process")
            if input_provided:
                logger.debug(f"U> {usr_in}")
            else:
                logger.debug("Sys> Enter 'r' for random selection, 'n' for skipping new image input")
                usr_in = input("U> ")

            try:
                if usr_in == "n":
                    logger.debug("Sys> Skipped image selection")
                    break
                elif usr_in == "r":
                    raise NotImplementedError       # Let's disable this for a while...
                    self.vision.new_input = self.dcompleter.sample()
                elif usr_in == "exit":
                    logger.debug("Sys> Terminating...")
                    quit()
                else:
                    # if usr_in not in self.dcompleter:
                    if not os.path.exists(usr_in):
                        raise ValueError(f"Image file {usr_in} does not exist")
                    self.vision.new_input = usr_in

            except ValueError as e:
                if input_provided:
                    raise e
                else:
                    logger.info(f"Sys> {e}, try again")

            else:
                self.vision.last_input = self.vision.new_input
                logger.info(f"Sys> Selected image file: {self.vision.new_input}")
                break

        # Restore default completer
        # readline.set_completer(rlcompleter.Completer().complete)
    
    def _lang_inp(self, usr_in=None):
        """Language input prompt (from user)"""
        self.lang.new_input = None
        input_provided = usr_in is not None

        logger.debug("Sys> Awaiting user input...")
        logger.debug("Sys> Enter 'n' for skipping language input")

        while True:
            if input_provided:
                # User input may be a single string or list of strings; if a single
                # string, wrap it into a singleton list. In current design, each usr_in
                # list represents a sequence of utterances in a single dialogue turn.
                if isinstance(usr_in, str):
                    usr_in = [usr_in]
                logger.info(f"U> {' '.join(usr_in)}")
            else:
                usr_in = input("U> ")

            try:
                if usr_in == "n":
                    logger.debug("Sys> Skipped language input")
                    break
                elif usr_in == "exit":
                    logger.debug("Sys> Terminating...")
                    quit()
                else:
                    self.lang.new_input = self.lang.semantic.nl_parse(usr_in)
                    break

            except IndexError as e:
                if input_provided:
                    raise e
                else:
                    logger.info(f"Sys> Ungrammatical input or IndexError: {e.args}")
            except ValueError as e:
                if input_provided:
                    raise e
                else:
                    logger.info(f"Sys> {e.args[0]}")

    def _update_belief(self, pointing=None):
        """ Form beliefs based on visual and/or language input """

        if not (self.vision.new_input or self.lang.new_input):
            # No information whatsoever to make any belief updates
            logger.debug("A> (Idling the moment away...)")
            return

        # Lasting storage of pointing info
        if pointing is None:
            pointing = {}

        # For showing visual UI on only the first time
        vis_ui_on = self.vis_ui_on

        # Index of latest dialogue turn
        ti_last = len(self.lang.dialogue.record)

        # Set of new visual concepts (equivalently, neologisms) newly registered
        # during the loop
        novel_concepts = set()

        # Recursive helper methods for checking whether rule head/body is grounded
        # (variable-free) or lifted (all variables)
        is_grounded = lambda cnj: all(not is_var for _, is_var in cnj.args) \
            if isinstance(cnj, Literal) else all(is_grounded(nc) for nc in cnj)
        is_lifted = lambda cnj: all(is_var for _, is_var in cnj.args) \
            if isinstance(cnj, Literal) else all(is_lifted(nc) for nc in cnj)

        # Keep updating beliefs until there's no more immediately exploitable learning
        # opportunities
        xb_updated = False      # Whether learning happened at neural-level (in exemplar base)
        kb_updated = False      # Whether learning happened at symbolic-level (in knowledge base)
        while True:
            ###################################################################
            ##                  Processing perceived inputs                  ##
            ###################################################################

            if self.vision.new_input is not None or xb_updated:
                # Ground raw visual perception with scene graph generation module
                self.vision.predict(
                    self.vision.last_input, self.lt_mem.exemplars,
                    visualize=vis_ui_on, lexicon=self.lt_mem.lexicon
                )
                vis_ui_on = False

            if self.vision.new_input is not None:
                # Inform the language module of the visual context
                self.lang.situate(self.vision.last_input, self.vision.scene)
                self.symbolic.refresh()

                # Reset below on episode-basis
                self.kb_snap = copy.deepcopy(self.lt_mem.kb.entries_by_pred)

            # Understand the user input in the context of the dialogue
            if self.lang.new_input is not None and self.lang.new_input[0]["raw"] != "Correct.":
                self.lang.dialogue.record = self.lang.dialogue.record[:ti_last]
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
                            "bbox_mode": "xyxy"
                        }
                        for ent in new_ents
                    }

                    # Incrementally predict on the designated bbox
                    self.vision.predict(
                        None, self.lt_mem.exemplars, bboxes=bboxes, visualize=False
                    )

            ###################################################################
            ##       Sensemaking via synthesis of perception+knowledge       ##
            ###################################################################

            dialogue_state = self.lang.dialogue.export_as_dict()

            if self.vision.new_input is not None or xb_updated or kb_updated:
                # Sensemaking from vision input only
                exported_kb = self.lt_mem.kb.export_reasoning_program()
                self.symbolic.sensemake_vis(self.vision.scene, exported_kb)

            if self.lang.new_input is not None:
                # Reference & word sense resolution to connect vision & discourse
                self.symbolic.resolve_symbol_semantics(
                    dialogue_state, self.lt_mem.lexicon
                )

                if self.vision.scene is not None:
                    # Sensemaking from vision & language input
                    self.symbolic.sensemake_vis_lang(dialogue_state)

            ###################################################################
            ##           Identify & exploit learning opportunities           ##
            ###################################################################

            # Resetting flags
            xb_updated = False
            kb_updated = False

            # Generic statements to be added to KB
            generics = []

            # Info needed (along with generics) for computing scalar implicatures
            pairRules = defaultdict(list)

            # Process translated dialogue record to do the following:
            #   - Integrate newly provided generic rules into KB
            #   - Identify recognition mismatch btw. user provided vs. agent
            #   - Identify visual concept confusion
            translated = self.symbolic.translate_dialogue_content(dialogue_state)
            for speaker, turn_clauses in translated:
                if speaker != "U": continue

                for (rule, _), raw in turn_clauses:
                    if rule is None: continue

                    head, body = rule
                    rule_is_grounded = (head is None or is_grounded(head)) and \
                        (body is None or is_grounded(body))
                    rule_is_lifted = (head is None or is_lifted(head)) and \
                        (body is None or is_lifted(body))

                    # Grounded event statement; test against vision-only sensemaking
                    # result to identify any mismatch btw. agent's & user's perception
                    # of world state
                    if rule_is_grounded and self.symbolic.concl_vis is not None:
                        # Make a yes/no query to obtain the likelihood of content
                        bjt_v, _ = self.symbolic.concl_vis
                        q_response = self.symbolic.query(bjt_v, None, rule)
                        ev_prob = q_response[()]

                        surprisal = -math.log(ev_prob + EPS)
                        if surprisal > -math.log(SR_THRES):
                            m = (rule, surprisal)
                            if m not in self.symbolic.mismatches:
                                self.symbolic.mismatches.append(m)

                    # Grounded fact; test against vision module output to identify 
                    # any 'concept overlap' -- i.e. whenever the agent confuses two
                    # concepts difficult to distinguish visually and mistakes one
                    # for another. Applicable only to experiment configs with maxHelp
                    # teachers.
                    if (rule_is_grounded and body is None and
                        self.cfg.exp1.strat_feedback == "maxHelp"):

                        for lit in head:
                            # Disregard negated conjunctions
                            if not isinstance(lit, Literal): continue

                            # (Temporary?) Only consider 1-place predicates, so retrieve
                            # the single and first entity from the arg list
                            conc_type, conc_ind = lit.name.split("_")
                            conc_ind = int(conc_ind)
                            ent = lit.args[0][0]

                            # (Temporary) Only consider non-part concepts as potential
                            # cases of confusion. This may be enforced in a more elegant
                            # way in the future..
                            cls_probs = self.vision.scene[ent]["pred_classes"][self.cls_offset:]

                            if ((conc_ind, conc_type) not in novel_concepts and
                                len(cls_probs) >= 2):
                                # Highest-score concept prediction for the entity of interest,
                                # and prediction score for true label
                                best_ind = np.argmax(cls_probs)
                                true_conc_score = cls_probs[conc_ind-self.cls_offset]

                                confusion_pair = frozenset(
                                    [best_ind+self.cls_offset, conc_ind]
                                )       # Potential confusion case, as unordered label pair

                                if (best_ind+self.cls_offset != conc_ind and
                                    true_conc_score >= SC_THRES and
                                    ("cls", confusion_pair) not in self.confused_no_more):
                                    # Agent's best guess disagrees with the user-provided
                                    # information, and score gap is small
                                    self.vision.confusions.add(("cls", confusion_pair))

                    # Symbolic knowledge base expansion; for generic rules without
                    # constant terms. Integrate the rule into KB by adding (for now
                    # we won't worry about intra-KB consistency, belief revision, etc.)
                    if rule_is_lifted:
                        generics.append((rule, U_IN_PR, raw))

                        if self.strat_generic != "semOnly":
                            # Current rule head conjunction & body conjunction as list
                            occurring_preds = {lit.name for lit in head+body}

                            # If agent's strategy of understanding generic statement
                            # is to exploit dialogue context (specifically, in the
                            # presence of record of a concept difference question)
                            agent_Qs = [
                                ques
                                for spk, turn_clauses in translated
                                for (_, ques), _ in turn_clauses
                                if spk == "A" and ques is not None
                            ]
                            diff_Qs = [
                                q_head[0] for q_vars, (q_head, q_body) in agent_Qs
                                if any(
                                    l.name=="*_diff" and l.args[2][0]==q_vars[0][0]
                                    for l in q_head
                                )
                            ]

                            if len(diff_Qs) > 0:
                                # Fetch two concepts being compared in the latest
                                # concept diff question
                                c1 = diff_Qs[-1].args[0][0]
                                c2 = diff_Qs[-1].args[1][0]
                                # Note: more principled way to manage relevant (I)QAP pair
                                # utterances would be to adopt a legitimate, established
                                # formalism for representing discourse structure (e.g. SDRT)

                                # (Ordered) Concept pair with which implicatures will be
                                # computed
                                if c1 in occurring_preds and c2 not in occurring_preds:
                                    rel_conc_pair = (c1, c2)
                                elif c2 in occurring_preds and c1 not in occurring_preds:
                                    rel_conc_pair = (c2, c1)
                                else:
                                    rel_conc_pair = None
                            else:
                                rel_conc_pair = None

                            if rel_conc_pair is not None:
                                # Compute appropriate implicatures for the concept pairs
                                # found
                                c1, c2 = rel_conc_pair

                                # Negative implicature; replace occurrence of c1 with c2
                                # then negate head conjunction (i.e. move head conj to body)
                                head_repl = tuple(
                                    l.substitute(preds={ c1: c2 }) for l in head
                                )
                                body_repl = tuple(
                                    l.substitute(preds={ c1: c2 }) for l in body
                                )
                                negImpl = ((list(head_repl),), body_repl)
                                generics.append((negImpl, A_IM_PR, f"{raw} (Neg. Impl.)"))

                                # Collect explicit generics provided for the concept pair
                                # and negative implicature computed from the context, with
                                # which scalar implicatures will be computed
                                pairRules[frozenset(rel_conc_pair)] += [
                                    rule, negImpl
                                ]

            # Update knowledge base with obtained generic statements
            for rule, w_pr, provenance in generics:
                kb_updated |= self.lt_mem.kb.add(
                    rule, w_pr, provenance
                )
            
            # Recursive helper method for substituting predicates while preserving
            # structure
            _substitute = lambda cnj, ps: cnj.substitute(preds=ps) \
                if isinstance(cnj, Literal) else [_substitute(nc, ps) for nc in cnj]

            # Scalar implicature; infer implicit concept similarities by copying
            # properties for c1/c2 and replacing the predicates with c2/c1, unless
            # the properties are denied by rules of "higher precedence level"
            if self.strat_generic == "semNegScal":
                # Helper method factored out for symmetric applications
                def computeScalarImplicature(c1, c2, rules):
                    # Return boolean flag indicating whether KB was updated
                    kb_updated = False

                    # Existing properties of c1
                    for i in self.kb_snap[c1]:
                        # Fetch KB entry
                        (head, body), _ = self.lt_mem.kb.entries[i]

                        # Replace occurrences of c1 with c2
                        head = tuple(_substitute(h, { c1: c2 }) for h in head)
                        body = tuple(_substitute(b, { c1: c2 }) for b in body)

                        # Negation of the replaced copy
                        if all(isinstance(h, Literal) for h in head):
                            # Positive conjunction head
                            head_neg = (list(head),)
                        elif all(isinstance(h, list) for h in head) and len(head)==1:
                            # Negated conjunction head
                            head_neg = tuple(head[0])
                        else:
                            # Cannot handle cases with rule head that is mixture
                            # of positive literals and negated conjunctions
                            raise NotImplementedError
                        
                        # Test the negated copy against explicitly stated generics
                        # and their negative implicature counterparts; they take
                        # precedence over defeasible implicatures
                        defeated = any(
                            Literal.isomorphic_conj_pair((head_neg, body), r)
                            for r in rules
                        )

                        if not defeated:
                            # Add the inferred generic that successfully survived
                            # the test against the higher-precedence rules
                            kb_updated |= self.lt_mem.kb.add(
                                (head, body), A_IM_PR, f"{c1} ~= {c2} (Scal. Impl.)"
                            )
                    
                    return kb_updated

                for (c1, c2), rules in pairRules.items():
                    kb_updated |= computeScalarImplicature(c1, c2, rules)
                    kb_updated |= computeScalarImplicature(c2, c1, rules)

            # Handle neologisms
            neologisms = {
                tok: sym for tok, (sym, den) in self.symbolic.word_senses.items()
                if den is None
            }
            for tok, sym in neologisms.items():
                neo_in_rule_head = tok[2] == "rh"
                neos_in_same_rule_body = [
                    n for n in neologisms if tok[:3]==n[:3] and n[3].startswith("b")
                ]
                if neo_in_rule_head and len(neos_in_same_rule_body)==0:
                    # Occurrence in rule head implies either definition or exemplar is
                    # provided by the utterance containing this token... Register new
                    # visual concept, and perform few-shot learning if appropriate
                    pos, name = sym
                    if pos == "n":
                        conc_type = "cls"
                    elif pos == "a":
                        conc_type = "att"
                    else:
                        assert pos == "v" or pos == "r"
                        conc_type = "rel"

                    # Expand corresponding visual concept inventory
                    conc_ind = self.vision.add_concept(conc_type)
                    novel_concept = (conc_ind, conc_type)
                    novel_concepts.add(novel_concept)

                    # Acquire novel concept by updating lexicon
                    self.lt_mem.lexicon.add((name, pos), novel_concept)

                    ti = int(tok[0].strip("t"))
                    ci = int(tok[1].strip("c"))
                    rule_head, rule_body = dialogue_state["record"][ti][1][ci][0][0]

                    if len(rule_body) == 0:
                        # Labelled exemplar provided; add new concept exemplars to
                        # memory, as feature vectors at the penultimate layer right
                        # before category prediction heads
                        args = [
                            self.symbolic.value_assignment[arg] for arg in rule_head[0][2]
                        ]
                        ex_bboxes = [
                            box_convert(
                                torch.tensor(self.lang.dialogue.referents["env"][a]["bbox"]),
                                "xyxy", "xywh"
                            ).numpy()
                            for a in args
                        ]

                        if conc_type == "cls":
                            f_vec = self.vision.f_vecs[args[0]][0]
                        elif conc_type == "att":
                            f_vec = self.vision.f_vecs[args[0]][1]
                        else:
                            assert conc_type == "rel"
                            raise NotImplementedError   # Step back for relation prediction...
                        
                        pointers_src = { 0: (0, tuple(ai for ai in range(len(args)))) }
                        pointers_exm = { conc_ind: ({0}, set()) }

                        self.lt_mem.exemplars.add_exs(
                            sources=[(np.asarray(self.vision.last_input), ex_bboxes)],
                            f_vecs={ conc_type: f_vec[None,:] },
                            pointers_src={ conc_type: pointers_src },
                            pointers_exm={ conc_type: pointers_exm }
                        )

                        # Set flag that XB is updated
                        xb_updated = True
                else:
                    # Otherwise not immediately resolvable
                    self.lang.unresolved_neologisms.add((sym, tok))

            # Terminate the loop when 'equilibrium' is reached
            if not (xb_updated or kb_updated):
                break

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

        for ti, ci in self.lang.dialogue.unanswered_Q:
            self.practical.agenda.append(("address_unanswered_Q", (ti, ci)))
        for n in self.lang.unresolved_neologisms:
            self.practical.agenda.append(("address_neologism", n))
        for m in self.symbolic.mismatches:
            self.practical.agenda.append(("address_mismatch", m))
        for c in self.vision.confusions:
            self.practical.agenda.append(("address_confusion", c))

        return_val = []

        while True:
            resolved_items = []
            for i, todo in enumerate(self.practical.agenda):
                todo_state, todo_args = todo

                # Check if this item can be resolved at this stage and if so, obtain
                # appropriate plan (sequence of actions) for resolving the item
                plan = self.practical.obtain_plan(todo_state)

                if plan is not None:
                    # Perform plan actions
                    for action in plan:
                        act_method = action["action_method"].extract(self)
                        act_args = action["action_args_getter"](todo_args)
                        if type(act_args) == tuple:
                            act_args = tuple(a.extract(self) for a in act_args)
                        else:
                            act_args = (act_args.extract(self),)

                        act_out = act_method(*act_args)
                        if act_out is not None:
                            return_val += act_out

                    resolved_items.append(i)

            if len(resolved_items) == 0:
                # No resolvable agenda item any more
                if (len(return_val) == 0 and self.lang.new_input is not None and
                    self.lang.new_input[0]["raw"] != "Correct."):
                    # Nothing to add, acknowledge any user input
                    self.practical.agenda.append(("acknowledge", None))
                else:
                    # Break with return vals
                    break
            else:
                # Check off resolved agenda item
                resolved_items.reverse()
                for i in resolved_items:
                    del self.practical.agenda[i]

        # Act; in our scope only available actions are dialogue utterance generation
        for act_type, act_data in return_val:
            if act_type == "generate":
                logger.info(f"A> {act_data}")

        return return_val
