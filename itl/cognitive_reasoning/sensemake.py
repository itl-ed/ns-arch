"""
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
import math
import copy

import numpy as np

from ..lpmln import Program, Rule, Literal
from ..lpmln.utils import wrap_args


EPS = 1e-10          # Value used for numerical stabilization
U_W_PR = 1.0         # How much the agent values information provided by the user
SR_THRES = -math.log(0.5)     # Surprisal threshold
TAB = "\t"           # For use in format strings

def sensemake_vis(vis_scene, objectness_thresh=0.75, category_thresh=0.75):
    """
    Combine raw visual perception outputs from the vision module (predictions with confidence)
    with existing knowledge to make final verdicts on the state of affairs, 'all things considered'.

    Args:
        vis_scene: Predictions (scene graphs) from the vision module
        objectness_thresh: float; Only consider recognised instances with objectness score
            higher than this value
        category_thresh: float; Only consider recognised categories with category score higher
            than this value

    Returns:
        List of possible worlds (models) with associated likelihood estimates,
        List of marginal probabilities for each grounded first-order literal,
        Composed ASP program (as string) used for generating these outputs,
        Rule counts
    """
    #########################################
    ## TODO: Add knowledgebase integration ##
    #########################################
    prog = Program()

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
    models_v, memoized_v = prog.solve()

    return models_v, memoized_v, prog


def sensemake_vis_lang(vis_result, dialogue_state, lexicon):
    """
    Combine raw visual perception outputs from the vision module (predictions with confidence)
    and the current dialogue information state with existing knowledge to make final verdicts
    on the state of affairs, 'all things considered'. Expects return values from sensemake_vis()
    as the first argument. Also, recognize and return any mismatches between the perceived scene
    graph versus the information contained in the understood dialogue state.

    Args:
        vis_result: Output from sensemaking with visual perception input only; return value from
            sensemake_vis() above
        dialogue_state: Current dialogue information state exported from the dialogue manager
        lexicon: Agent's lexicon, required for matching between environment entities vs. discourse
            referents for variable assignment

    Returns:
        List of possible worlds (models) with associated likelihood estimates,
        List of marginal probabilities for each grounded first-order literal,
        Composed ASP program (as string) used for generating these outputs,
        + Any mismatches between vis-only vs. vis-and-lang marginals
    """
    models_v, memoized_v, prog = vis_result

    # Incorporate info from dialogue state
    aprog = Program()      # For finding referent-entity assignment
    dprog = Program()      # For incorporating additional info from user
    
    # Find the best estimate of assignment

    # Environment entities
    marginals_v = models_v.marginals()
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
            # won't require the full processing implemented below

            if body is not None:
                body_pos = [bl for bl in body if not bl[3]]
                body_neg = [bl for bl in body if bl[3]]

                if len(body_pos) > 0:
                    # Penalize assignments satisfying the positive part of rule body (which could
                    # be relieved later if the rule head exists and is satisfied)
                    a_body = []
                    for j, bl in enumerate(body_pos):
                        b_args = bl[2] + (f"{bl[1]}_{bl[0]}", f"Wb{j}")
                        a_body.append(Literal("hold", wrap_args(*b_args)))

                        # Consult lexicon for word sense selection
                        vis_concepts = lexicon.s2d[(bl[0], bl[1])]
                        aprog.add_rule(Rule(
                            head=[
                                Literal("denote", wrap_args(f"{bl[1]}_{bl[0]}", f"{vc[1]}_{vc[0]}"))
                                for vc in vis_concepts
                            ],
                            lb=1, ub=1
                        ))

                    W_summ = [op for j in range(len(body_pos)) for op in (f"Wb{str(j)}", "+")][:-1]
                    aprog.add_hard_rule(Rule(
                        head=Literal("pen", [(f"r{i}_bp", False), (W_summ, True)]),
                        body=a_body
                    ))

                if len(body_neg) > 0:
                    # Lessen penalties for assignments satisfying the negative part of rule body, so
                    # that assignments fulfilling the whole body is penalized the most
                    a_body = []
                    for j, bl in enumerate(body_neg):
                        b_args = bl[2] + (f"{bl[1]}_{bl[0]}", f"Wb{j}")
                        a_body.append(Literal("hold", wrap_args(*b_args)))

                        # Consult lexicon for word sense selection
                        vis_concepts = lexicon.s2d[(bl[0], bl[1])]
                        aprog.add_rule(Rule(
                            head=[
                                Literal("denote", wrap_args(f"{bl[1]}_{bl[0]}", f"{vc[1]}_{vc[0]}"))
                                for vc in vis_concepts
                            ],
                            lb=1, ub=1
                        ))

                    W_summ = ["-"]+[op for j in range(len(body_neg)) for op in (f"Wb{str(j)}", "-")][:-1]
                    aprog.add_hard_rule(Rule(
                        head=Literal("pen", [(f"r{i}_bn", False), (W_summ, True)]),
                        body=a_body
                    ))

            if head is not None:
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
                        Literal("denote", wrap_args(f"{head[1]}_{head[0]}", f"{vc[1]}_{vc[0]}"))
                        for vc in vis_concepts
                    ],
                    lb=1, ub=1
                ))

                hb_rule = Rule(
                    head=Literal("pen", [(f"r{i}_hb", False), (W_summ, True)]),
                    body=a_body
                )
                aprog.add_hard_rule(hb_rule)

    # Hard assignments by pointing, etc.
    for ref, env in dialogue_state["assignment_hard"].items():
        aprog.add_hard_rule(
            Rule(body=[Literal("assign", [(ref, False), (env, False)], naf=True)])
        )

    ## Assignment program rules

    # hold(X,PL,W) :- hold(E,PV,W), assign(X,E), denote(PL, PV).
    aprog.add_hard_rule(Rule(
        head=Literal("hold", wrap_args("X", "PL", "W")),
        body=[
            Literal("hold", wrap_args("E", "PV", "W")),
            Literal("assign", wrap_args("X", "E")),
            Literal("denote", wrap_args("PL", "PV"))
        ]
    ))
    # hold(X1,X2,PL,W) :- hold(E1,E2,PV,W), assign(X1,E1), assign(X2,E2), denote(PL, PV).
    aprog.add_hard_rule(Rule(
        head=Literal("hold", wrap_args("X1", "X2", "P", "W")),
        body=[
            Literal("hold", wrap_args("E1", "E2", "P", "W")),
            Literal("assign", wrap_args("X1", "E1")),
            Literal("assign", wrap_args("X2", "E2")),
            Literal("denote", wrap_args("PL", "PV"))
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
        ("minimize", [
            ([Literal("zero_p", [])], "0", []),
            ([Literal("pen", wrap_args("RI", "W"))], "W", ["RI"])
        ])
    ])

    best_assignment = [atom.args for atom in opt_models[0] if atom.name == "assign"]
    best_assignment = {args[0][0]: args[1][0] for args in best_assignment}

    word_senses = [atom.args for atom in opt_models[0] if atom.name == "denote"]
    word_senses = {
        tuple(symbol[0].split("_")): tuple(denotation[0].split("_"))
        for symbol, denotation in word_senses
    }
    word_senses = {
        (symbol[1], symbol[0]): (denotation[0], denotation[1])
        for symbol, denotation in word_senses.items()
    }

    a_map = lambda args: [best_assignment[a] for a in args]

    # Now incorporate additional information provided by the user in language for updated
    # sensemaking
    subs_rules = []
    for _, _, (info, _), _ in dialogue_state["record"]:
        for i, rule in enumerate(info):
            head, body, _ = rule

            if head is not None:
                pred = "_".join(word_senses[head[:2]])
                args = [a for a in a_map(head[2])]
                subs_head = Literal(pred, wrap_args(*args))
            else:
                subs_head = None
            
            if body is not None:
                subs_body = []
                for bl in body:
                    pred = "_".join(word_senses[bl[:2]])
                    args = [a for a in a_map(bl[2])]
                    bl = Literal(pred, wrap_args(*args), naf=bl[3])
                    subs_body.append(bl)
            else:
                subs_body = None

            subs_rule = Rule(head=subs_head, body=subs_body)
            subs_rules.append(subs_rule)
            dprog.add_rule(subs_rule, U_W_PR)

    # Finally, reasoning with all visual+language info
    prog += dprog
    models_vl, memoized_vl = prog.solve(provided_mem=memoized_v)

    # Identify any drastic mismatches between vis only vs. lang input
    mismatches = []

    # Test provided info contained in dialogue record against vision-only cognition
    for rule in subs_rules:
        ev_prob = models_v.query_yn(subs_rule)

        surprisal = -math.log(ev_prob + EPS)
        if surprisal > SR_THRES:
            mismatches.append((True, subs_rule, surprisal))

    return (models_vl, memoized_vl, prog), best_assignment, mismatches
