"""
Cognitive reasoning module API that exposes only the high-level functionalities
required by the ITL agent: make sense out of the current visual & language inputs
plus existing knowledge stored in knowledge base
"""
from collections import defaultdict

from .sensemake import sensemake_vis, sensemake_vis_lang


class CognitiveReasonerModule:

    def __init__(self, kb):
        self.kb = kb
        self.concl_vis = None
        self.concl_vis_lang = None
        self.Q_answers = defaultdict(list)
    
    def refresh(self):
        self.concl_vis = None
        self.concl_vis_lang = None
        self.Q_answers = defaultdict(list)

    def sensemake(self, vision, lang, practical):
        """
        Put together all information available to make final conclusion regarding the
        current state of affairs
        """
        vis_scene = vision.vis_scene
        dialogue_state = lang.export_dialogue_state()
        lexicon = lang.lexicon
        agenda = practical.agenda

        self.concl_vis = sensemake_vis(vis_scene)

        if len(dialogue_state["record"]) > 0:
            self.concl_vis_lang, assignment, mismatches = sensemake_vis_lang(
                self.concl_vis, dialogue_state, lexicon
            )

            lang.update_referent_assignment(assignment)

            ## TODO: Print surprisal reports to resolve_mismatch action ##
            if len(mismatches) > 0:
                # print("A> However, I was quite surprised to hear that:")

                for m in mismatches:
                    # is_positive = m[0]     # 'True' stands for positive statement

                    # if is_positive:
                    #     # Positive statement (fact)
                    #     message = f"{m[2][2][0]} is (a/an) {m[1][0]}"
                    # else:
                    #     # Negative statement (constraint)
                    #     negated = [
                    #         f"{m2[2][0]} is (a/an) {m1[0]}" for m1, m2 in zip(m[1], m[2])
                    #     ]
                    #     message = f"Not {{{' & '.join(negated)}}}"

                    # print(f"A> {TAB}{message} (surprisal: {round(m[3], 3)})")
                
                    agenda.append(("unresolved_mismatch", m))

    def compute_answer_to_Q(self, utt_id, lang):
        """ Compute answers to question indexed by indexed by utt_id """
        rules, queries = lang.utt_to_ASP(utt_id)
        current_models = self.concl_vis_lang[0]

        for q_ent, q_lits, _ in queries:

            if q_ent is None:
                # Y/N questions
                to_query = set()

                for ql in q_lits:
                    pr_split = ql.name.split("_")

                    if pr_split[0] == "*":
                        # Special reserved predicates
                        if pr_split[1] == "=":
                            # (For now we will assume arg1 is definite, and arg2 is not. TODO: Relax
                            # the assumption.)
                            # When arg1 is definite and arg2 is not, we can replace occurrences of
                            # arg2 in rules with arg1 to see if they hold, effectively performing
                            # object identity check between the definite arg1 and some supposed arg2
                            # satisfying its provided specifications.
                            for r, _ in rules:
                                to_query |= {
                                    hl.substitute(ql.args[1][0], ql.args[0][0]) for hl in r.head
                                }
                        else:
                            raise ValueError(f"Unrecognized special predicate: {pr_split[1]}")

                    else:
                        # Query the Models instance from the sensemaking result
                        for r, _ in rules:
                            to_query |= set(r.head)
                        to_query.add(ql)

                # Answer to a Y/N question is a probability value (0 ~ 1); the degree to
                # which the queried statement is likely to be true, according to current
                # understanding of situation.
                ans = current_models.query_yn(to_query)

            else:
                # Wh- questions
                ans = ...
        
            self.Q_answers[utt_id].append(ans)
