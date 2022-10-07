"""
theoretical reasoning module API that exposes only the high-level functionalities
required by the ITL agent: make sense out of the current visual & language inputs
plus existing knowledge stored in knowledge base.

Implements 'sensemaking' process; that is, process of integrating and re-organizing
perceived information from various modalities to establish a set of judgements structured
in such a manner that they can later be exploited for other symbolic reasoning tasks --
in light of the existing general knowledge held by the perceiving agent.

(I borrow the term 'sensemaking' from the discipline of theoretical science & psychology.
According to Klein (2006), sensemaking is "the process of creating situational awareness
and understanding in situations of high complexity or uncertainty in order to make decisions".)

Here, we resort to declarative programming to encode individual sensemaking problems into
logic programs (written in the language of weighted ASP) and solve with a dedicated solver
(clingo).
"""
from itertools import product
from collections import defaultdict

import numpy as np

from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import wrap_args


TAB = "\t"              # For use in format strings

EPS = 1e-10             # Value used for numerical stabilization
U_IN_PR = 1.0           # How much the agent values information provided by the user
SCORE_THRES = 0.5      # Only consider recognised categories with category score higher
                        # than this value, unless focused attention warranted by KB
LOWER_THRES = 0.3       # Lower threshold for predicates that deserve closer look

class TheoreticalReasonerModule:

    def __init__(self):
        self.concl_vis = None
        self.concl_vis_lang = None
        self.Q_answers = {}

        self.value_assignment = {}    # Store best assignments (tentative) obtained by reasoning
        self.word_senses = {}         # Store current estimate of symbol denotations

        self.mismatches = set()

    def refresh(self):
        self.concl_vis = None
        self.concl_vis_lang = None
        self.Q_answers = {}

        self.value_assignment = {}
        self.word_senses = {}

        self.mismatches = set()

    def sensemake_vis(self, vis_scene, exported_kb):
        """
        Combine raw visual perception outputs from the vision module (predictions with
        confidence) with existing knowledge to make final verdicts on the state of affairs,
        'all things considered'.

        Args:
            vis_scene: Predictions (scene graphs) from the vision module
            exported_kb: Output from KnowledgeBase().export_reasoning_program()
        """
        inference_prog, preds_in_kb = exported_kb

        # Build ASP program for processing perception outputs
        pprog = Program()

        # Biasing literals by visual evidence
        for oi, obj in vis_scene.items():
            # Object classes
            if "pred_classes" in obj:
                classes = set(np.where(obj["pred_classes"] > SCORE_THRES)[0])
                classes |= preds_in_kb["cls"] & \
                    set(np.where(obj["pred_classes"] > LOWER_THRES)[0])
                for c in classes:
                    # p_vis ::  cls_C(oi).
                    event_lit = Literal(f"cls_{c}", [(oi,False)])
                    rule = Rule(head=event_lit)
                    r_pr = float(obj["pred_classes"][c])

                    pprog.add_rule(rule, r_pr)

            # Object attributes
            if "pred_attributes" in obj:
                attributes = set(np.where(obj["pred_attributes"] > SCORE_THRES)[0])
                attributes |= preds_in_kb["att"] & \
                    set(np.where(obj["pred_attributes"] > LOWER_THRES)[0])
                for a in attributes:
                    # p_vis ::  att_A(oi).
                    event_lit = Literal(f"att_{a}", [(oi,False)])
                    rule = Rule(head=event_lit)
                    r_pr = float(obj["pred_attributes"][a])

                    pprog.add_rule(rule, r_pr)

            # Object relations
            if "pred_relations" in obj:
                relations = {
                    oj: set(np.where(per_obj > SCORE_THRES)[0]) | \
                        (preds_in_kb["rel"] & set(np.where(per_obj > LOWER_THRES)[0]))
                    for oj, per_obj in obj["pred_relations"].items()
                }
                for oj, per_obj in relations.items():
                    for r in per_obj:
                        # p_vis ::  rel_R(oi,oj).
                        event_lit = Literal(f"rel_{r}", [(oi,False),(oj,False)])
                        rule = Rule(head=event_lit)
                        r_pr = float(obj["pred_relations"][oj][r])

                        pprog.add_rule(rule, r_pr)

        # Solve with clingo to find the best models of the program
        prog = pprog + inference_prog
        models_v = prog.solve()

        # Store sensemaking result as module state
        self.concl_vis = models_v, prog
    
    def resolve_symbol_semantics(self, dialogue_state, lexicon):
        """
        Find a fully specified mapping from symbols in discourse record to corresponding
        entities and concepts; in other words, perform reference resolution and word
        sense disambiguation.

        Args:
            dialogue_state: Current dialogue information state exported from the dialogue
                manager
            lexicon: Agent's lexicon, required for matching between environment entities
                vs. discourse referents for variable assignment
        """
        # Find the best estimate of referent value assignment
        aprog = Program()

        # Environmental referents
        occurring_atoms = set()
        for ent in dialogue_state["referents"]["env"]:
            if ent not in occurring_atoms:
                aprog.add_absolute_rule(Rule(head=Literal("env", wrap_args(ent))))
                occurring_atoms.add(ent)

        # Discourse referents
        for rf, v in dialogue_state["referents"]["dis"].items():
            if not (v["is_univ_quantified"] or v["is_wh_quantified"]):
                aprog.add_absolute_rule(Rule(head=Literal("dis", wrap_args(rf))))
                if v["is_referential"]:
                    aprog.add_absolute_rule(Rule(head=Literal("referential", wrap_args(rf))))

        # Hard assignments by pointing, etc.
        for ref, env in dialogue_state["assignment_hard"].items():
            aprog.add_absolute_rule(Rule(head=Literal("env", wrap_args(env))))
            aprog.add_absolute_rule(
                Rule(body=[Literal("assign", [(ref, False), (env, False)], naf=True)])
            )

        # Add priming effect by recognized visual concepts
        if self.concl_vis is not None:
            models_v, _ = self.concl_vis
            ## TODO: uncomment after updating compute_marginals
            # marginals_v = models_v.compute_marginals()

            # if marginals_v is not None:
            #     vis_concepts = defaultdict(float)
            #     for atom, pr in marginals_v.items():
            #         is_cls = atom.name.startswith("cls")
            #         is_att = atom.name.startswith("att")
            #         is_rel = atom.name.startswith("rel")
            #         if is_cls or is_att or is_rel:
            #             # Collect 'priming intensity' by recognized concepts
            #             vis_concepts[atom.name] += pr[0]

            #     for vc, score in vis_concepts.items():
            #         aprog.add_absolute_rule(Rule(
            #             head=Literal("vis_prime", wrap_args(vc, int(score * 100)))
            #         ))

        # Understood dialogue record contents
        occurring_preds = set()
        for ui, (_, (rules, question), _) in enumerate(dialogue_state["record"]):
            if rules is not None:
                for ri, rule in enumerate(rules):
                    head, body, _ = rule

                    head_preds = [] if head is None else [h[:2] for h in head]
                    body_preds = [] if body is None else [b[:2] for b in body]
                    occurring_preds |= set(head_preds+body_preds)

                    # Symbol token occurrence locations
                    for c, preds in [("h", head_preds), ("b", body_preds)]:
                        for pi, p in enumerate(preds):
                            # Skip special reserved predicates
                            if p[1] == "*": continue

                            sym = f"{p[1]}_{p[0].split('/')[0]}"
                            tok_loc = f"u{ui}_r{ri}_{c}{pi}"
                            aprog.add_absolute_rule(
                                Rule(head=Literal("pred_token", wrap_args(tok_loc, sym)))
                            )

                    head_args = set() if head is None else set(sum([h[2] for h in head], ()))
                    body_args = set() if body is None else set(sum([b[2] for b in body], ()))
                    occurring_args = head_args | body_args

                    if all(a in dialogue_state["assignment_hard"] for a in occurring_args):
                        # Below not required if all occurring args are hard-assigned to some entity
                        continue

                    # If models_v is present and rule is grounded, add bias in favor of
                    # assignments which would satisfy the rule
                    is_grounded = all(not a[0].isupper() for a in occurring_args)
                    if self.concl_vis is not None and is_grounded:
                        # Rule instances by possible word sense selections
                        wsd_cands = [lexicon.s2d[sym[:2]] for sym in head_preds+body_preds]

                        for vcs in product(*wsd_cands):
                            if head is not None:
                                head_vcs = vcs[:len(head_preds)]
                                rule_head = [
                                    Literal(f"{vc[1]}_{vc[0]}", wrap_args(*h[2]), naf=h[3])
                                    for h, vc in zip(head, head_vcs)
                                ]
                            else:
                                rule_head = None

                            if body is not None:
                                body_vcs = vcs[len(head_preds):]
                                rule_body = [
                                    Literal(f"{vc[1]}_{vc[0]}", wrap_args(*b[2]), naf=b[3])
                                    for b, vc in zip(body, body_vcs)
                                ]
                            else:
                                rule_body = None

                            # Question to query models_v with
                            q_rule = Rule(head=rule_head, body=rule_body)
                            q_vars = tuple((a, False) for a in occurring_args)
                            query_result, _ = models_v.query(q_vars, q_rule)

                            for ans, (_, score) in query_result.items():
                                c_head = Literal("cons", wrap_args(f"r{ri}", int(score*100)))
                                c_body = [
                                    Literal("assign", wrap_args(x[0], e)) for x, e in zip(q_vars, ans)
                                ]
                                if head is not None:
                                    c_body += [
                                        Literal("denote", wrap_args(f"u{ui}_r{ri}_h{hi}", f"{d[1]}_{d[0]}"))
                                        for hi, d in enumerate(head_vcs)
                                    ]
                                if body is not None:
                                    c_body += [
                                        Literal("denote", wrap_args(f"u{ui}_r{ri}_b{bi}", f"{d[1]}_{d[0]}"))
                                        for bi, d in enumerate(body_vcs)
                                    ]

                                aprog.add_absolute_rule(Rule(head=c_head, body=c_body))
            
            if question is not None:
                _, q_rules = question

                for qi, rule in enumerate(q_rules):
                    head, body, _ = rule

                    head_preds = [] if head is None else [h[:2] for h in head]
                    body_preds = [] if body is None else [b[:2] for b in body]
                    occurring_preds |= set(head_preds+body_preds)

                    # Symbol token occurrence locations
                    for c, preds in [("h", head_preds), ("b", body_preds)]:
                        for pi, p in enumerate(preds):
                            # Skip special reserved predicates
                            if p[1] == "*": continue

                            sym = f"{p[1]}_{p[0].split('/')[0]}"
                            tok_loc = f"u{ui}_q{qi}_{c}{pi}"
                            aprog.add_absolute_rule(
                                Rule(head=Literal("pred_token", wrap_args(tok_loc, sym)))
                            )

        # Predicate info needed for word sense selection
        for p in occurring_preds:
            # Skip special reserved predicates
            if p[1] == "*": continue

            sym = f"{p[1]}_{p[0].split('/')[0]}"

            # Consult lexicon to list denotation candidates
            if p in lexicon.s2d:
                for vc in lexicon.s2d[p]:
                    pos_match = (p[1], vc[1]) == ("n", "cls") \
                        or (p[1], vc[1]) == ("a", "att") \
                        or (p[1], vc[1]) == ("r", "rel") \
                        or (p[1], vc[1]) == ("v", "rel")
                    if not pos_match: continue

                    den = f"{vc[1]}_{vc[0]}"

                    aprog.add_absolute_rule(
                        Rule(head=Literal("may_denote", wrap_args(sym, den)))
                    )
                    aprog.add_absolute_rule(
                        Rule(head=Literal("d_freq", wrap_args(den, lexicon.d_freq[vc])))
                    )
            else:
                # Predicate symbol not found in lexicon: unresolved neologism
                aprog.add_absolute_rule(
                    Rule(head=Literal("may_denote", wrap_args(sym, "_neo")))
                )

        ## Assignment program rules

        # 1 { assign(X,E) : env(E) } 1 :- dis(X), referential(X).
        aprog.add_rule(Rule(
            head=Literal(
                "assign", wrap_args("X", "E"),
                conds=[Literal("env", wrap_args("E"))]
            ),
            body=[
                Literal("dis", wrap_args("X")),
                Literal("referential", wrap_args("X"))
            ],
            lb=1, ub=1
        ))

        # { assign(X,E) : env(E) } 1 :- dis(X), not referential(X).
        aprog.add_rule(Rule(
            head=Literal(
                "assign", wrap_args("X", "E"),
                conds=[Literal("env", wrap_args("E"))]
            ),
            body=[
                Literal("dis", wrap_args("X")),
                Literal("referential", wrap_args("X"), naf=True)
            ],
            ub=1
        ))

        # 1 { denote(T,D) : may_denote(S,D) } 1 :- pred_token(T,S).
        aprog.add_rule(Rule(
            head=Literal(
                "denote", wrap_args("T", "D"),
                conds=[Literal("may_denote", wrap_args("S", "D"))]
            ),
            body=[
                Literal("pred_token", wrap_args("T", "S"))
            ],
            lb=1, ub=1
        ))

        # 'Base cost' for cases where no assignments are any better than others
        aprog.add_absolute_rule(Rule(head=Literal("zero_p", [])))

        # By querying for the optimal assignment, essentially we are giving the user a 'benefit
        # of doubt', such that any statements made by the user are considered as True, and the
        # agent will try to find the 'best' assignment to make it so.
        # (Note: this is not a probabilistic inference, and the confidence scores provided as 
        # arguments are better understood as properties of the env. entities & disc. referents.)
        opt_models = aprog.optimize([
            # (Note: Earlier statements receive higher optimization priority)
            # Prioritize assignments that agree with given statements
            ("maximize", [
                ([Literal("zero_p", [])], "0", []),
                ([Literal("cons", wrap_args("RI", "S"))], "S", ["RI"])
            ]),
            # Prioritize word senses that occur in visual scene (if any): 'priming effect'
            ("maximize", [
                ([
                    Literal("denote", wrap_args("T", "D")),
                    Literal("vis_prime", wrap_args("D", "S"))
                ], "S", ["T"])
            ]),
            # Prioritize word senses with higher frequency
            ("maximize", [
                ([
                    Literal("denote", wrap_args("T", "D")),
                    Literal("d_freq", wrap_args("D", "F"))
                ], "F", ["T"])
            ])
        ])

        best_assignment = [atom.args for atom in opt_models[0] if atom.name == "assign"]
        best_assignment = {args[0][0]: args[1][0] for args in best_assignment}

        tok2sym_map = [atom.args[:2] for atom in opt_models[0] if atom.name == "pred_token"]
        tok2sym_map = {
            tuple(token[0].split("_")): tuple(symbol[0].split("_"))
            for token, symbol in tok2sym_map
        }

        word_senses = [atom.args[:2] for atom in opt_models[0] if atom.name == "denote"]
        word_senses = {
            tuple(token[0].split("_")): denotation[0]
            for token, denotation in word_senses
        }
        word_senses = {
            token: (tok2sym_map[token], denotation)
                if denotation != "_neo" else (tok2sym_map[token], None)
            for token, denotation in word_senses.items()
        }

        self.value_assignment.update(best_assignment)
        self.word_senses.update(word_senses)

    def translate_dialogue_content(self, dialogue_state):
        """
        Translate logical content of dialogue record (which should be already ASP-
        compatible) into ASP (LP^MLN) rules/queries, based on current estimate of
        value assignment and word sense selection. Dismiss (replace with None) any
        utterances containing unresolved neologisms.
        """
        result = []
        a_map = lambda args: [self.value_assignment.get(a, a) for a in args]

        for ui, (_, (rules, question), _) in enumerate(dialogue_state["record"]):
            # If the utterance contains an unresolved neologism, give up translation
            # for the time being
            contains_unresolved_neologism = any([
                den is None for tok, (_, den) in self.word_senses.items()
                if tok[0]==f"u{ui}"
            ])
            if contains_unresolved_neologism:
                result.append((None, None))
                continue

            # Translate rules
            if rules is not None:
                translated_rules = []
                for ri, rule in enumerate(rules):
                    head, body, _ = rule

                    if head is not None:
                        rule_head = [
                            Literal(
                                self.word_senses[(f"u{ui}",f"r{ri}",f"h{hi}")][1],
                                args=wrap_args(*a_map(h[2])), naf=h[3]
                            )
                            for hi, h in enumerate(head)
                        ]
                    else:
                        rule_head = None

                    if body is not None:
                        rule_body = [
                            Literal(
                                self.word_senses[(f"u{ui}",f"r{ri}",f"b{bi}")][1],
                                args=wrap_args(*a_map(b[2])), naf=b[3]
                            )
                            for bi, b in enumerate(body)
                        ]
                    else:
                        rule_body = None

                    translated_rules.append(Rule(head=rule_head, body=rule_body))
            else:
                translated_rules = None
            
            # Translate question
            if question is not None:
                q_vars, q_rules = question

                translated_qrs = []
                for qi, (head, body, _) in enumerate(q_rules):
                    if head is not None:
                        rule_head = [
                            # If head literal predicate is not found in self.word_senses,
                            # it means the predicate is a special reserved one
                            Literal(
                                self.word_senses[(f"u{ui}",f"q{qi}",f"h{hi}")][1],
                                args=wrap_args(*a_map(h[2])), naf=h[3]
                            )
                            if (f"u{ui}",f"q{qi}",f"h{hi}") in self.word_senses
                            else Literal(
                                "_".join(h[1::-1]), args=wrap_args(*a_map(h[2])), naf=h[3]
                            )
                            for hi, h in enumerate(head)
                        ]
                    else:
                        rule_head = None

                    if body is not None:
                        rule_body = [
                            Literal(
                                self.word_senses[(f"u{ui}",f"q{qi}",f"b{bi}")][1],
                                args=wrap_args(*a_map(b[2])), naf=b[3]
                            )
                            for bi, b in enumerate(body)
                        ]
                    else:
                        rule_body = None

                    translated_qrs.append(Rule(head=rule_head, body=rule_body))

                translated_question = q_vars, translated_qrs
            else:
                translated_question = None

            result.append((translated_rules, translated_question))

        return result

    def sensemake_vis_lang(self, dialogue_state):
        """
        Combine raw visual perception outputs from the vision module (predictions with
        confidence) and the current dialogue information state with existing knowledge
        to make final verdicts on the state of affairs, 'all things considered'.

        Args:
            dialogue_state: Current dialogue information state exported from the dialogue
                manager
        """
        dprog = Program()
        models_v, prog = self.concl_vis

        # Incorporate additional information provided by the user in language for updated
        # sensemaking
        for rules, _ in self.translate_dialogue_content(dialogue_state):
            if rules is not None:
                for r in rules:
                    # Skip any non-grounded content
                    head_has_var = len(r.head) > 0 and any([
                        any(is_var for _, is_var in h.args) for h in r.head
                    ])
                    body_has_var = len(r.body) > 0 and any([
                        any(is_var for _, is_var in b.args) for b in r.body
                    ])
                    if head_has_var or body_has_var: continue

                    dprog.add_rule(r, U_IN_PR)

        # Finally, reasoning with all visual+language info
        if len(dprog) > 0:
            prog += dprog
            models_vl = prog.solve()
        else:
            models_vl = models_v

        # Store sensemaking result as module state
        self.concl_vis_lang = models_vl, prog
