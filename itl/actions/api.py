"""
Agent actions API that implements and exposes 'composite' actions that require
interplay between more than one agent modules, internal or external. Actions are
to be registered to an ITLAgent instance provided as __init__ arg first (thus
forming circular reference), and later evoked by plans fetched from the practical
reasoning module.
"""
import random
from functools import reduce
from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_convert

from ..lpmln import Literal, Rule


class AgentCompositeActions:
    
    def __init__(self, agent):
        """
        Args:
            agent: ITLAgent instance that will perform the actions
        """
        self.agent = agent

    def handle_mismatch(self, mismatch):
        """
        Handle recognition gap following some specified strategy. Note that we now
        assume the user (teacher) is an infallible oracle, and the agent doesn't
        question info provided from user.
        """
        # This mismatch is about to be handled
        self.agent.symbolic.mismatches.remove(mismatch)

        rule, _ = mismatch

        if self.agent.cfg.agent.strat_mismatch == "zeroInit":
            # Zero initiative from agent's end; do not ask any further question, simply
            # perform few-shot vision model updates (if possible) and acknowledge "OK"
            if rule.is_grounded() and len(rule.literals())==1:
                if rule.is_fact():
                    # Positive grounded fact
                    atom = rule.head[0]
                    exm_pointer = ({0}, set())
                else:
                    # Negative grounded fact
                    atom = rule.body[0]
                    exm_pointer = (set(), {0})

                conc_type, conc_ind = atom.name.split("_")
                conc_ind = int(conc_ind)
                args = [a for a, _ in atom.args]

                ex_bboxes = [
                    box_convert(
                        torch.tensor(self.agent.lang.dialogue.referents["env"][a]["bbox"]),
                        "xyxy", "xywh"
                    ).numpy()
                    for a in args
                ]

                # Fetch current score for the asserted fact
                if conc_type == "cls":
                    f_vec = self.agent.vision.f_vecs[args[0]][0]
                elif conc_type == "att":
                    f_vec = self.agent.vision.f_vecs[args[0]][1]
                else:
                    assert conc_type == "rel"
                    raise NotImplementedError   # Step back for relation prediction...

                # Add new concept exemplars to memory, as feature vectors at the
                # penultimate layer right before category prediction heads
                pointers_src = { 0: (0, tuple(ai for ai in range(len(args)))) }
                pointers_exm = { conc_ind: exm_pointer }

                self.agent.lt_mem.exemplars.add_exs(
                    sources=[(np.asarray(self.agent.vision.last_input), ex_bboxes)],
                    f_vecs={ conc_type: f_vec[None,:] },
                    pointers_src={ conc_type: pointers_src },
                    pointers_exm={ conc_type: pointers_exm }
                )

                # This now shouldn't strike the agent as surprise, at least in this
                # loop (Ideally this doesn't need to be enforced this way, if the
                # few-shot learning capability is perfect)
                self.agent.doubt_no_more.add(rule)

            if ("acknowledge", None) not in self.agent.practical.agenda:
                self.agent.practical.agenda.append(("acknowledge", None))

        elif self.agent.cfg.agent.strat_mismatch == "request_exmp":
            raise NotImplementedError

        else:
            assert self.agent.cfg.agent.strat_mismatch == "request_expl"
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
        dialogue_state = self.agent.lang.dialogue.export_as_dict()
        translated = self.agent.symbolic.translate_dialogue_content(dialogue_state)

        _, question = translated[ui]

        if question is None:
            # Question cannot be answered for some reason
            return
        else:
            # Schedule to answer the question
            self.agent.practical.agenda.append(("answer_Q", ui))
            return

    def prepare_answer_Q(self, ui):
        """
        Prepare an answer to a question that has been deemed answerable, by first computing
        raw ingredients from which answer candidates can be composed, picking out an answer
        among the candidates, then translating the answer into natural language form to be
        uttered
        """
        # The question is about to be answered
        self.agent.lang.dialogue.unanswered_Q.remove(ui)

        dialogue_state = self.agent.lang.dialogue.export_as_dict()
        _, _, orig_utt = dialogue_state["record"][ui]

        translated = self.agent.symbolic.translate_dialogue_content(dialogue_state)

        _, question = translated[ui]
        assert question is not None

        q_vars, _ = question
        bjt_vl, _ = self.agent.symbolic.concl_vis_lang

        # Ensure it has every ingredient available needed for making most informed judgements
        # on computing the best answer to the question. Specifically, scene graph outputs from
        # vision module may be omitting some entities, whose presence and properties may have
        # critical influence on the symbolic sensemaking process. Make sure such entities, if
        # actually present, are captured in scene graphs by performing visual search as needed.
        if len(self.agent.lt_mem.kb.entries) > 0:
            search_specs = self._search_specs_from_kb(question, bjt_vl)
            if len(search_specs) > 0:
                self.agent.vision.predict(
                    None, self.agent.lt_mem.exemplars,
                    specs=search_specs, visualize=False, lexicon=self.agent.lt_mem.lexicon
                )

                #  ... and another round of sensemaking
                exported_kb = self.agent.lt_mem.kb.export_reasoning_program()
                self.agent.symbolic.sensemake_vis(self.agent.vision.scene, exported_kb)
                self.agent.symbolic.resolve_symbol_semantics(dialogue_state, self.agent.lt_mem.lexicon)
                self.agent.symbolic.sensemake_vis_lang(dialogue_state)

                bjt_vl, _ = self.agent.symbolic.concl_vis_lang

        # Compute raw answer candidates by appropriately querying compiled BJT
        answers_raw = self.agent.symbolic.query(bjt_vl, *question)

        if q_vars is not None:
            # (Temporary) Enforce non-part concept as answer. This may be enforced in a more
            # elegant way in the future...
            for ans in list(answers_raw.keys()):
                if not(ans[0] == "cls_11" or ans[0] == "cls_12" or ans[0] == "cls_13"):
                    del answers_raw[ans]

        # Pick out an answer to deliver; maximum confidence
        if len(answers_raw) > 0:
            max_score = max(answers_raw.values())
            answer_selected = random.choice([
                a for (a, s) in answers_raw.items() if s == max_score
            ])
            ev_prob = answers_raw[answer_selected]
        else:
            answer_selected = (None,) * len(q_vars)
            ev_prob = None

        # Translate the selected answer into natural language
        # (Parse the original question utterance, manipulate, then generate back)
        if len(answer_selected) == 0:
            # Yes/no question; cast original question to proposition
            answer_translated = self.agent.lang.semantic.nl_change_sf(orig_utt, "prop")

            if ev_prob < 0.5:
                # Flip polarity for negative answer with event probability lower than 0.5
                answer_translated = self.agent.lang.semantic.nl_negate(answer_translated)
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
                        self.agent.lang.dialogue.to_generate.append("I am not sure.")
                        return
                    else:
                        ans = ans.split("_")
                        ans = (int(ans[1]), ans[0])

                        is_named = False
                        val = self.agent.lt_mem.lexicon.d2s[ans][0][0]
                else:
                    # Entity by their constant name handle
                    is_named = True
                    if low_confidence:
                        val = None
                    else:
                        val = ans

                replace_values.append((val, is_named))

            # Plug in the selected answer in place of the wh-quantified referent
            answer_translated = self.agent.lang.semantic.nl_replace_wh(
                answer_translated, replace_targets, replace_values
            )

            # Don't forget to make it a prop
            answer_translated = self.agent.lang.semantic.nl_change_sf(answer_translated, "prop")

        # Push the translated answer to buffer of utterances to generate
        self.agent.lang.dialogue.to_generate.append(answer_translated)

    def _search_specs_from_kb(self, question, ref_bjt):
        """
        Factored helper method for extracting specifications for visual search,
        based on the agent's current knowledge-base entries and some sensemaking
        result provided as a compiled binary join tree (BJT)
        """
        q_vars, q_rules = question

        # Queries (in IR sense) to feed into KB for fetching search specs. Represent each
        # query as a pair of predicates of interest & arg entities of interest
        kb_queries = set()

        for qr in q_rules:
            # Inspecting literals in each q_rule for identifying search specs to feed into
            # visual search calls
            for q_lit in qr.literals():
                if q_lit.name == "*_?":
                    # Literal whose predicate is question-marked (contained for questions
                    # like "What is this?", etc.); the first argument term, standing for
                    # the predicate variable, must be contained in q_vars
                    assert q_lit.args[0] in q_vars

                    # Assume we are only interested in cls concepts with "What is this?"
                    # type of questions
                    kb_query_preds = frozenset([
                        pred for pred in self.agent.lt_mem.kb.entries_by_pred 
                        if pred.startswith("cls")
                    ])
                    # (Temporary) Enforce non-part concept as answer. This may be enforced in a more
                    # elegant way in the future...
                    kb_query_preds = frozenset([
                        pred for pred in kb_query_preds
                        if pred == "cls_11" or pred == "cls_12" or pred == "cls_13"
                    ])

                    kb_query_args = tuple(q_lit.args[1:])
                else:
                    # Literal with fixed predicate, to which can narrow down the KB query
                    kb_query_preds = frozenset([q_lit.name])
                    kb_query_args = tuple(q_lit.args)
                
                kb_queries.add((kb_query_preds, kb_query_args))

        # Query the KB to collect search specs
        search_spec_cands = []
        for kb_qr in kb_queries:
            kb_query_preds, kb_query_args = kb_qr

            for pred in kb_query_preds:
                # Relevant KB entries containing predicate of interest
                relevant_entries = self.agent.lt_mem.kb.entries_by_pred[pred]
                relevant_entries = [
                    self.agent.lt_mem.kb.entries[entry_id]
                    for entry_id in relevant_entries
                ]

                # Set of literals for each relevant KB entry
                relevant_literals = [
                    set.union(*[r.literals() for r in entry[0]])
                    for entry in relevant_entries
                ]
                # Depending on which literal (with matching predicate name) in literal
                # sets to use as 'anchor', there can be multiple choices of search specs
                relevant_literals = [
                    { l: lits-{l} for l in lits if l.name==pred }
                    for lits in relevant_literals
                ]

                # Collect search spec candidates. We will disregard attribute concepts as
                # search spec elements, noticing that it is usually sufficient and generalizable
                # to provide object class info only as specs for searching potentially relevant,
                # yet unrecognized entities in a scene. This is more of a heuristic for now --
                # maybe justify this on good grounds later...
                specs = [
                    {
                        tgt_lit: (
                            {rl for rl in rel_lits if not rl.name.startswith("att_")},
                            {la: qa for la, qa in zip(tgt_lit.args, kb_query_args)}
                        )
                        for tgt_lit, rel_lits in lits.items()
                    }
                    for lits in relevant_literals
                ]
                specs = [
                    {
                        tgt_lit.substitute(terms=term_map): frozenset({
                            rl.substitute(terms=term_map) for rl in rel_lits
                        })
                        for tgt_lit, (rel_lits, term_map) in spc.items()
                    }
                    for spc in specs
                ]
                search_spec_cands += specs

        # Merge and flatten down to a single layer dict
        def set_add_merge(d1, d2):
            for k, v in d2.items(): d1[k].add(v)
            return d1
        search_spec_cands = reduce(set_add_merge, [defaultdict(set)]+search_spec_cands)

        # Finalize set of search specs, excluding those which already have satisfying
        # entities in the current sensemaking output
        final_specs = []
        for lits_sets in search_spec_cands.values():
            for lits in lits_sets:
                # Lift any remaining function term args to non-function variable args
                all_fn_args = {
                    a for a in set.union(*[set(l.args) for l in lits])
                    if type(a[0])==tuple
                }
                all_var_names = {
                    t_val for t_val, t_is_var in set.union(*[l.nonfn_terms() for l in lits])
                    if t_is_var
                }
                fn_lifting_map = {
                    fa: (f"X{i+len(all_var_names)}", True)
                    for i, fa in enumerate(all_fn_args)
                }

                search_vars = all_var_names | {vn for vn, _ in fn_lifting_map.values()}
                search_vars = tuple(search_vars)
                if len(search_vars) == 0:
                    # Disregard if there's no variables in search spec (i.e. no search target
                    # after all)
                    continue

                lits = [l.substitute(terms=fn_lifting_map) for l in lits]
                lits = frozenset([l for l in lits if any(la_is_var for _, la_is_var in l.args)])

                # Disregard if there's already an isomorphic literal set
                has_isomorphic_spec = any(
                    Literal.isomorphism_btw(lits, spc[1], None) is not None
                    for spc in final_specs
                )
                if has_isomorphic_spec:
                    continue

                # Check if the agent is already (visually) aware of the potential search
                # targets; if so, disregard this one
                check_result = self.agent.symbolic.query(
                    ref_bjt,
                    tuple((v, False) for v in search_vars),
                    frozenset([Rule(head=l) for l in lits])
                )
                if len(check_result) > 0:
                    continue

                final_specs.append((search_vars, lits))

        # Perform incremental visual search...
        O = len(self.agent.vision.scene)
        oi_offsets = np.cumsum([0]+[len(vars) for vars, _ in final_specs][:-1])
        final_specs = {
            tuple(f"o{offset+i+O}" for i in range(len(spc[0]))): spc
            for spc, offset in zip(final_specs, oi_offsets)
        }

        return final_specs
