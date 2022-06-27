"""
Recognitive reasoning module API that exposes only the high-level functionalities
required by the ITL agent: make sense out of the current visual & language inputs
plus existing knowledge stored in knowledge base.

Implements 'sensemaking' process; that is, process of integrating and re-organizing
perceived information from various modalities to establish a set of judgements structured
in such a manner that they can later be exploited for other symbolic reasoning tasks --
in light of the existing general knowledge held by the perceiving agent.

(I borrow the term 'sensemaking' from the discipline of recognitive science & psychology.
According to Klein (2006), sensemaking is "the process of creating situational awareness
and understanding in situations of high complexity or uncertainty in order to make decisions".)

Here, we resort to declarative programming to encode individual sensemaking problems into
logic programs (written in the language of weighted ASP) and solve with a dedicated solver
(clingo).
"""
import copy
from itertools import product
from collections import defaultdict

import numpy as np

from ..lpmln import Rule, Program, Literal
from ..lpmln.utils import wrap_args


EPS = 1e-10          # Value used for numerical stabilization
U_W_PR = 1.0         # How much the agent values information provided by the user
TAB = "\t"           # For use in format strings

class RecognitiveReasonerModule:

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

    def sensemake_vis(self, vis_scene, kb_prog, objectness_thresh=0.75, category_thresh=0.75):
        """
        Combine raw visual perception outputs from the vision module (predictions with
        confidence) with existing knowledge to make final verdicts on the state of affairs,
        'all things considered'.

        Args:
            vis_scene: Predictions (scene graphs) from the vision module
            objectness_thresh: float; Only consider recognised instances with objectness
                score higher than this value
            category_thresh: float; Only consider recognised categories with category
                score higher than this value
        """
        prog = kb_prog

        # Filter by objectness threshold (recognized objects are already ranked by
        # objectness score)
        vis_scene = copy.deepcopy({
            oi: obj for oi, obj in vis_scene.items()
            if obj["pred_objectness"] > objectness_thresh
        })
        # Accordingly exclude per-object relation predictions
        for obj in vis_scene.values():
            obj["pred_relations"] = {
                oi: per_obj for oi, per_obj in obj["pred_relations"].items()
                if oi in vis_scene
            }

        # Build ASP program for processing perception outputs
        pprog = Program()
        classes_all = set(); attributes_all = set(); relations_all = set()

        for oi, obj in vis_scene.items():
            # Objectness
            rule = Rule(head=Literal("object", [(oi,False)]))
            w_pr = float(obj["pred_objectness"])
            pprog.add_rule(rule, w_pr)

            # Object classes
            classes = set(np.where(obj["pred_classes"] > category_thresh)[0])
            if len(obj["pred_classes"]) > 0:
                # Also add the max category even if score is below threshold
                classes.add(obj["pred_classes"].argmax())
            classes_all |= classes
            for c in classes:
                rule = Rule(
                    head=Literal(f"cls_{c}", [(oi,False)]),
                    body=[Literal("object", [(oi,False)])]
                )
                w_pr = float(obj["pred_classes"][c])
                pprog.add_rule(rule, w_pr)

            # Object attributes
            attributes = set(np.where(obj["pred_attributes"] > category_thresh)[0])
            if len(obj["pred_attributes"]) > 0:
                attributes.add(obj["pred_attributes"].argmax())
            attributes_all |= attributes
            for a in attributes:
                rule = Rule(
                    head=Literal(f"att_{a}", [(oi,False)]),
                    body=[Literal("object", [(oi,False)])]
                )
                w_pr = float(obj["pred_attributes"][a])
                pprog.add_rule(rule, w_pr)
            
            # Object relations
            relations = {
                oj: set(np.where(per_obj > category_thresh)[0])
                for oj, per_obj in obj["pred_relations"].items()
            }
            for oj, per_obj in relations.items():
                if len(obj["pred_relations"][oj]) > 0:
                    per_obj.add(obj["pred_relations"][oj].argmax())
                relations_all |= per_obj
                for r in per_obj:
                    rule = Rule(
                        head=Literal(f"rel_{r}", [(oi,False),(oj,False)]),
                        body=[
                            Literal("object", [(oi,False)]),
                            Literal("object", [(oj,False)])
                        ]
                    )
                    w_pr = float(obj["pred_relations"][oj][r])
                    pprog.add_rule(rule, w_pr)

        # Solve with clingo to find the best K_M models of the program
        prog += pprog
        if self.concl_vis is not None:
            _, memoized_v, _ = self.concl_vis
        else:
            memoized_v = None
        models_v, memoized_v = prog.solve(provided_mem=memoized_v)

        # Store sensemaking result as module state
        self.concl_vis = models_v, memoized_v, prog
    
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

        # Environmental entities & recognized visual concepts
        if self.concl_vis is not None:
            models_v, _, _ = self.concl_vis
            marginals_v, _ = models_v.marginals()

            vis_concepts = defaultdict(float)
            for atom, pr in marginals_v.items():
                if atom.name == "object":
                    aprog.add_hard_rule(Rule(head=Literal("env", atom.args)))
                else:
                    # Collect 'priming intensity' by recognized concepts
                    vis_concepts[atom.name] += pr
            
            for vc, score in vis_concepts.items():
                aprog.add_hard_rule(Rule(
                    head=Literal("vis_prime", wrap_args(vc, int(score * 100)))
                ))

        # Discourse referents
        for rf, v in dialogue_state["referents"]["dis"].items():
            if not (v["is_univ_quantified"] or v["is_wh_quantified"]):
                aprog.add_hard_rule(Rule(head=Literal("dis", wrap_args(rf))))
                if v["is_referential"]:
                    aprog.add_hard_rule(Rule(head=Literal("referential", wrap_args(rf))))

        # Hard assignments by pointing, etc.
        for ref, env in dialogue_state["assignment_hard"].items():
            aprog.add_hard_rule(Rule(head=Literal("env", wrap_args(env))))
            aprog.add_hard_rule(
                Rule(body=[Literal("assign", [(ref, False), (env, False)], naf=True)])
            )

        # Understood dialogue record contents
        occurring_preds = set()
        for ui, (_, _, (rules, query), _) in enumerate(dialogue_state["record"]):
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

                            sym = f"{p[1]}_{p[0]}"
                            tok_loc = f"u{ui}_r{ri}_{c}{pi}"
                            aprog.add_hard_rule(
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

                            # Rule to query models_v with
                            q_rule = Rule(head=rule_head, body=rule_body)
                            q_vars = tuple(occurring_args)
                            query_result, _ = models_v.query(q_vars, q_rule)

                            for ans, (_, score) in query_result.items():
                                c_head = Literal("cons", wrap_args(f"r{ri}", int(score*100)))
                                c_body = [
                                    Literal("assign", wrap_args(x, e)) for x, e in zip(q_vars, ans)
                                ] + [
                                    Literal("denote", wrap_args(f"u{ui}_r{ri}_p{pi}", f"{d[1]}_{d[0]}"))
                                    for pi, d in enumerate(vcs)
                                ]

                                aprog.add_hard_rule(Rule(head=c_head, body=c_body))
            
            if query is not None:
                _, q_rules = query

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

                            sym = f"{p[1]}_{p[0]}"
                            tok_loc = f"u{ui}_q{qi}_{c}{pi}"
                            aprog.add_hard_rule(
                                Rule(head=Literal("pred_token", wrap_args(tok_loc, sym)))
                            )

        # Predicate info needed for word sense selection
        for p in occurring_preds:
            # Skip special reserved predicates
            if p[1] == "*": continue

            # Consult lexicon to list denotation candidates
            sym = f"{p[1]}_{p[0]}"
            if p in lexicon.s2d:
                for vc in lexicon.s2d[p]:
                    den = f"{vc[1]}_{vc[0]}"
                    aprog.add_hard_rule(
                        Rule(head=Literal("may_denote", wrap_args(sym, den)))
                    )
                    aprog.add_hard_rule(
                        Rule(head=Literal("d_freq", wrap_args(den, lexicon.d_freq[vc])))
                    )
            else:
                # Predicate symbol not found in lexicon: unresolved neologism
                aprog.add_hard_rule(
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
        aprog.add_hard_rule(Rule(head=Literal("zero_p", [])))

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
            tuple(token[0].split("_")): tuple(denotation[0].split("_"))
            for token, denotation in word_senses
        }
        word_senses = {
            token: (tok2sym_map[token], (denotation[0], int(denotation[1])))
                if len(denotation[0])>0
                else (tok2sym_map[token], None)
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

        for ui, (_, _, (rules, query), _) in enumerate(dialogue_state["record"]):
            # If the utterance contains an unresolved neologism, give up translation
            # for the time being
            contains_unresolved_neologism = any([
                den[1] is None for tok, den in self.word_senses.items() if tok[0]==f"u{ui}"
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
                            self.word_senses[(f"u{ui}",f"r{ri}",f"h{hi}")][1]
                            for hi in range(len(head))
                        ]
                        rule_head = [
                            Literal(
                                f"{rule_head[i][0]}_{rule_head[i][1]}",
                                args=wrap_args(*a_map(h[2])), naf=h[3]
                            )
                            for i, h in enumerate(head)
                        ]
                    else:
                        rule_head = None

                    if body is not None:
                        rule_body = [
                            self.word_senses[(f"u{ui}",f"r{ri}",f"b{bi}")][1]
                            for bi in range(len(body))
                        ]
                        rule_body = [
                            Literal(
                                f"{rule_body[i][0]}_{rule_body[i][1]}",
                                args=wrap_args(*a_map(b[2])), naf=b[3]
                            )
                            for i, b in enumerate(body)
                        ]
                    else:
                        rule_body = None

                    translated_rules.append(Rule(head=rule_head, body=rule_body))
            else:
                translated_rules = None
            
            # Translate query
            if query is not None:
                q_vars, q_rules = query

                translated_qrs = []
                for qi, (head, body, _) in enumerate(q_rules):
                    if head is not None:
                        rule_head = [
                            # If head literal predicate is not found in self.word_senses,
                            # it means the predicate is a special reserved one
                            self.word_senses.get(
                                (f"u{ui}",f"q{qi}",f"h{hi}"), (None, head[hi][1::-1])
                            )[1]
                            for hi in range(len(head))
                        ]
                        rule_head = [
                            Literal(
                                f"{rule_head[i][0]}_{rule_head[i][1]}",
                                args=wrap_args(*a_map(h[2])), naf=h[3]
                            )
                            for i, h in enumerate(head)
                        ]
                    else:
                        rule_head = None

                    if body is not None:
                        rule_body = [
                            self.word_senses[(f"u{ui}",f"q{qi}",f"b{bi}")][1]
                            for bi in range(len(body))
                        ]
                        rule_body = [
                            Literal(
                                f"{rule_body[i][0]}_{rule_body[i][1]}",
                                args=wrap_args(*a_map(b[2])), naf=b[3]
                            )
                            for i, b in enumerate(body)
                        ]
                    else:
                        rule_body = None

                    translated_qrs.append(Rule(head=rule_head, body=rule_body))

                translated_query = q_vars, translated_qrs
            else:
                translated_query = None

            result.append((translated_rules, translated_query))

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
        models_v, memoized_v, prog = self.concl_vis

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

                    dprog.add_rule(r, U_W_PR)

        # Finally, reasoning with all visual+language info
        if len(dprog) > 0:
            prog += dprog
            models_vl, memoized_vl = prog.solve(provided_mem=memoized_v)
        else:
            models_vl, memoized_vl = models_v, memoized_v

        # Store sensemaking result as module state
        self.concl_vis_lang = models_vl, memoized_vl, prog
