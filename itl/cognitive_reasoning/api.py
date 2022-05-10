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

        if vision.updated:
            self.concl_vis = sensemake_vis(vis_scene)

        if len(dialogue_state["record"]) > 0:
            self.concl_vis_lang, (assignment, word_senses), mismatches = sensemake_vis_lang(
                self.concl_vis, dialogue_state, lexicon
            )

            lang.update_referent_assignment(assignment)
            lang.update_word_senses(word_senses)

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
        _, query = lang.utt_to_ASP(utt_id)
        current_models = self.concl_vis_lang[0]

        q_ents, q_rules, _ = query
        q_ans = current_models.query(q_ents, q_rules)
    
        self.Q_answers[utt_id].append(q_ans)
