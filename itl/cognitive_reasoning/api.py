"""
Cognitive reasoning module API that exposes only the high-level functionalities
required by the ITL agent: make sense out of the current visual & language inputs
plus existing knowledge stored in knowledge base.

Implements 'sensemaking' process; that is, process of integrating and re-organizing
perceived information from various modalities to establish a set of judgements structured
in such a manner that they can later be exploited for other symbolic reasoning tasks --
in light of the existing general knowledge held by the perceiving agent.

(I borrow the term 'sensemaking' from the discipline of cognitive science & psychology.
According to Klein (2006), sensemaking is "the process of creating situational awareness
and understanding in situations of high complexity or uncertainty in order to make decisions".)

Here, we resort to declarative programming to encode individual sensemaking problems into
logic programs (written in the language of weighted ASP) and solve with a dedicated solver
(clingo).
"""
import copy

import numpy as np

from ..lpmln import Rule, Program, Literal
from ..lpmln.utils import wrap_args


EPS = 1e-10          # Value used for numerical stabilization
U_W_PR = 1.0         # How much the agent values information provided by the user
TAB = "\t"           # For use in format strings

class CognitiveReasonerModule:

    def __init__(self):
        self.concl_vis = None
        self.concl_vis_lang = None
        self.Q_answers = {}

        self.value_assignment = {}   # Store best assignments (tentative) obtained by reasoning
        self.word_senses = {}          # Store current estimate of symbol denotations
    
    def refresh(self):
        self.concl_vis = None
        self.concl_vis_lang = None
        self.Q_answers = {}

        self.value_assignment = {}
        self.word_senses = {}

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
            classes.add(obj["pred_classes"].argmax())    # Also add the max category
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
        if len(dialogue_state["record"]) == 0:
            # Don't bother
            return

        # Find the best estimate of referent value assignment
        aprog = Program()

        if self.concl_vis is not None:
            models_v, _, _ = self.concl_vis

            # Environment entities
            marginals_v, _ = models_v.marginals()
            for atom, conf in marginals_v.items():
                pred = atom.name
                args = atom.args

                if pred == "object":
                    # Occurring entities
                    aprog.add_hard_rule(Rule(head=Literal("env", args)))

                else:
                    # Estimated marginals with probabilities
                    aprog.add_hard_rule(
                        Rule(head=Literal("hold", args+wrap_args(pred, str(int(100*float(conf))))))
                    )

        # Discourse referents: Consider information from dialogue state

        # Referent info
        for rf, value in dialogue_state["referents"]["dis"].items():
            aprog.add_hard_rule(Rule(head=Literal("dis", wrap_args(rf))))
            if value["is_referential"]:
                aprog.add_hard_rule(Rule(head=Literal("referential", wrap_args(rf))))

        # Understood dialogue record contents
        for _, _, (info, _), _ in dialogue_state["record"]:
            for i, rule in enumerate(info):
                head, body, _ = rule

                # The most general way of adding constraints from understood rule, though
                # most rules will be either body-less facts or head-less constraints and thus
                # won't require the full processing implemented below...

                if body is not None:
                    body_has_var = body is not None and any([
                        any([type(x)==str and x[0].isupper() for x in bl[2]])
                        for bl in body
                    ])
                    body_pos = [bl for bl in body if not bl[3]]
                    body_neg = [bl for bl in body if bl[3]]

                    if len(body_pos) > 0:
                        # Penalize assignments satisfying the positive part of rule body (which could
                        # be relieved later if the rule head exists and is satisfied)
                        a_body = []
                        for j, bl in enumerate(body_pos):
                            # Consult lexicon for word sense selection
                            vis_concepts = lexicon.s2d[(bl[0], bl[1])]
                            aprog.add_rule(Rule(
                                head=[
                                    Literal("denote", wrap_args(
                                        f"{bl[1]}_{bl[0]}", f"{vc[1]}_{vc[0]}", lexicon.d_freq[vc]
                                    ))
                                    for vc in vis_concepts
                                ],
                                lb=1, ub=1
                            ))

                            b_args = bl[2] + (f"{bl[1]}_{bl[0]}", f"Wb{j}")
                            a_body.append(Literal("hold", wrap_args(*b_args)))

                        if not body_has_var:
                            W_summ = [
                                op for j in range(len(body_pos)) for op in (f"Wb{str(j)}", "+")
                            ][:-1]
                            aprog.add_hard_rule(Rule(
                                head=Literal("pen", [(f"r{i}_bp", False), (W_summ, True)]),
                                body=a_body
                            ))

                    if len(body_neg) > 0:
                        # Lessen penalties for assignments satisfying the negative part of rule body,
                        # so that assignments fulfilling the whole body is penalized the most
                        a_body = []
                        for j, bl in enumerate(body_neg):
                            # Consult lexicon for word sense selection
                            vis_concepts = lexicon.s2d[(bl[0], bl[1])]
                            aprog.add_rule(Rule(
                                head=[
                                    Literal("denote", wrap_args(
                                        f"{bl[1]}_{bl[0]}", f"{vc[1]}_{vc[0]}", lexicon.d_freq[vc]
                                    ))
                                    for vc in vis_concepts
                                ],
                                lb=1, ub=1
                            ))

                            b_args = bl[2] + (f"{bl[1]}_{bl[0]}", f"Wb{j}")
                            a_body.append(Literal("hold", wrap_args(*b_args), naf=True))

                        if not body_has_var:
                            W_summ = str(len(body_neg) * 100)
                            aprog.add_hard_rule(Rule(
                                head=Literal("pen", [(f"r{i}_bn", False), (W_summ, True)]),
                                body=a_body
                            ))

                if head is not None:
                    head_has_var = head is not None and any([
                        type(x)==str and x[0].isupper() for x in head[2]
                    ])

                    # Reward assignments satisfying the rule head and body so that the potential penalties
                    # imposed above could be compensated
                    h_args = head[2] + (f"{head[1]}_{head[0]}", "Wh")
                    a_body = [Literal("hold", wrap_args(*h_args))]
                    W_summ = ["-", "Wh"]

                    if body is not None:
                        for j, bl in enumerate(body):
                            vis_concept = lexicon.s2d[(bl[0], bl[1])]
                            b_args = bl[2] + (f"{vis_concept[1]}_{vis_concept[0]}", f"Wb{j}")
                            a_body.append(Literal("hold", wrap_args(*b_args)))

                            if not bl[3]:
                                W_summ += ["-", f"Wb{j}"]

                    # Consult lexicon for word sense selection
                    vis_concepts = lexicon.s2d[(head[0], head[1])]
                    aprog.add_rule(Rule(
                        head=[
                            Literal("denote", wrap_args(
                                f"{head[1]}_{head[0]}", f"{vc[1]}_{vc[0]}", lexicon.d_freq[vc]
                            ))
                            for vc in vis_concepts
                        ],
                        lb=1, ub=1
                    ))

                    if not head_has_var:
                        hb_rule = Rule(
                            head=Literal("pen", [(f"r{i}_hb", False), (W_summ, True)]),
                            body=a_body
                        )
                        aprog.add_hard_rule(hb_rule)

        # Hard assignments by pointing, etc.
        for ref, env in dialogue_state["assignment_hard"].items():
            aprog.add_hard_rule(Rule(head=Literal("env", wrap_args(env))))
            aprog.add_hard_rule(
                Rule(body=[Literal("assign", [(ref, False), (env, False)], naf=True)])
            )

        ## Assignment program rules

        # hold(X,PS,W) :- hold(E,PD,W), assign(X,E), denote(PS, PD, F).
        aprog.add_hard_rule(Rule(
            head=Literal("hold", wrap_args("X", "PS", "W")),
            body=[
                Literal("hold", wrap_args("E", "PD", "W")),
                Literal("assign", wrap_args("X", "E")),
                Literal("denote", wrap_args("PS", "PD", "F"))
            ]
        ))
        # hold(X1,X2,PS,W) :- hold(E1,E2,PD,W), assign(X1,E1), assign(X2,E2), denote(PS, PD, F).
        aprog.add_hard_rule(Rule(
            head=Literal("hold", wrap_args("X1", "X2", "P", "W")),
            body=[
                Literal("hold", wrap_args("E1", "E2", "P", "W")),
                Literal("assign", wrap_args("X1", "E1")),
                Literal("assign", wrap_args("X2", "E2")),
                Literal("denote", wrap_args("PS", "PD", "F"))
            ]
        ))

        # 1 { assign(X,E) : env(E) } 1 :- dis(X), referential(X).
        aprog.add_rule(Rule(
            head=Literal(
                "assign", wrap_args("X", "E"), conds=[Literal("env", wrap_args("E"))]
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
                "assign", wrap_args("X", "E"), conds=[Literal("env", wrap_args("E"))]
            ),
            body=[
                Literal("dis", wrap_args("X")),
                Literal("referential", wrap_args("X"), naf=True)
            ],
            ub=1
        ))

        # 'Base cost' for cases where no assignments are any better than others
        aprog.add_hard_rule(Rule(head=Literal("zero_p", [])))

        # By querying for the optimal assignment, essentially we are giving the user a 'benefit
        # of doubt', such that any statements made by the user are considered as True, and the
        # agent will try to find the 'best' assignment to make it so.
        # (Note: this is not a probabilistic inference, and the confidence scores provided as 
        # arguments are better understood as properties of the env. entities & disc. referents.)
        opt_models = aprog.optimize([
            # Note: Earlier statements receive higher optimization priority
            ("minimize", [
                ([Literal("zero_p", [])], "0", []),
                ([Literal("pen", wrap_args("RI", "W"))], "W", ["RI"])
            ]),
            ("maximize", [
                ([Literal("denote", wrap_args("PS", "PD", "F"))], "F", ["PS"])
            ])
        ])

        best_assignment = [atom.args for atom in opt_models[0] if atom.name == "assign"]
        best_assignment = {args[0][0]: args[1][0] for args in best_assignment}

        word_senses = [atom.args[:2] for atom in opt_models[0] if atom.name == "denote"]
        word_senses = {
            tuple(symbol[0].split("_")): tuple(denotation[0].split("_"))
            for symbol, denotation in word_senses
        }
        word_senses = {
            (symbol[1], symbol[0]): (denotation[0], int(denotation[1]))
            for symbol, denotation in word_senses.items()
        }

        self.value_assignment.update(best_assignment)
        self.word_senses.update(word_senses)

    def sensemake_vis_lang(self, dialogue_state):
        """
        Combine raw visual perception outputs from the vision module (predictions with
        confidence) and the current dialogue information state with existing knowledge
        to make final verdicts on the state of affairs, 'all things considered'.

        Args:
            dialogue_state: Current dialogue information state exported from the dialogue
                manager
        """
        if len(dialogue_state["record"]) == 0:
            # Don't bother
            return

        assert self.concl_vis is not None

        dprog = Program()
        a_map = lambda args: [self.value_assignment[a] for a in args]

        _, memoized_v, prog = self.concl_vis

        # Incorporate additional information provided by the user in language for updated
        # sensemaking
        for _, _, (info, _), _ in dialogue_state["record"]:
            for i, rule in enumerate(info):
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
                    pred = "_".join([str(s) for s in self.word_senses[head[:2]]])
                    args = [a for a in a_map(head[2])]
                    subs_head = Literal(pred, wrap_args(*args))
                else:
                    subs_head = None
                
                if body is not None:
                    subs_body = []
                    for bl in body:
                        pred = "_".join(self.word_senses[bl[:2]])
                        args = [a for a in a_map(bl[2])]
                        bl = Literal(pred, wrap_args(*args), naf=bl[3])
                        subs_body.append(bl)
                else:
                    subs_body = None

                dprog.add_rule(Rule(head=subs_head, body=subs_body), U_W_PR)

        # Finally, reasoning with all visual+language info
        prog += dprog
        models_vl, memoized_vl = prog.solve(provided_mem=memoized_v)

        # Store sensemaking result as module state
        self.concl_vis_lang = models_vl, memoized_vl, prog
