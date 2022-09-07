"""
Agent actions API that implements and exposes 'composite' actions that require
interplay between more than one agent modules, internal or external. Actions are
to be registered to an ITLAgent instance first, and later evoked by plans fetched
from PracticalReasonerModule.
"""
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
        self.agent.theoretical.mismatches.remove(mismatch)

        rule, _ = mismatch

        if self.agent.opts.strat_mismatch == "zeroInit":
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

                cat_type, concept_ind = atom.name.split("_")
                concept_ind = int(concept_ind)
                args = [a for a, _ in atom.args]

                ex_bboxes = [
                    self.agent.lang.dialogue.referents["env"][a]["bbox"] for a in args
                ]

                # Fetch current score for the asserted fact
                if cat_type == "cls":
                    f_vec = self.agent.vision.f_vecs[0][args[0]]
                elif cat_type == "att":
                    f_vec = self.agent.vision.f_vecs[1][args[0]]
                else:
                    assert cat_type == "rel"
                    f_vec = self.agent.vision.f_vecs[2][args[0]][args[1]]

                # Add new concept exemplars to memory, as feature vectors at the
                # penultimate layer right before category prediction heads
                pointers_src = { 0: (0, tuple(ai for ai in range(len(args)))) }
                pointers_exm = { concept_ind: exm_pointer }

                self.agent.lt_mem.exemplars.add_exs(
                    sources=[(self.agent.vision.last_raw, ex_bboxes)],
                    f_vecs={ cat_type: f_vec[None,:].cpu().numpy() },
                    pointers_src={ cat_type: pointers_src },
                    pointers_exm={ cat_type: pointers_exm }
                )

                # Update the category code parameter in the vision model's predictor
                # head using the new set of exemplars
                imperfect_concept = (concept_ind, cat_type)
                self.agent.vision.update_concept(
                    imperfect_concept, self.agent.lt_mem.exemplars[imperfect_concept]
                )

                # This now shouldn't strike the agent as surprise, at least in this
                # loop (Ideally this doesn't need to be enforced this way, if the
                # few-shot learning capability is perfect)
                self.agent.doubt_no_more.add(rule)

            if ("acknowledge", None) not in self.agent.practical.agenda:
                self.agent.practical.agenda.append(("acknowledge", None))

        elif self.agent.opts.strat_mismatch == "request_exmp":
            raise NotImplementedError

        else:
            assert self.agent.opts.strat_mismatch == "request_expl"
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
        translated = self.agent.theoretical.translate_dialogue_content(dialogue_state)

        _, query = translated[ui]

        if query is None:
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

        translated = self.agent.theoretical.translate_dialogue_content(dialogue_state)

        _, query = translated[ui]
        assert query is not None

        q_vars, _ = query

        # Ensure it has every ingredient available for making most informed judgements
        # on computing the best answer to the question. Namely, for logic-based reasoner,
        # inspect its current belief after sensemaking against its KB, performing visual
        # search
        print(0)

        # Compute raw answer candidates by appropriately querying current world models
        models_vl, _, _ = self.agent.theoretical.concl_vis_lang
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
