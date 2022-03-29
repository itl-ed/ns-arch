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
from collections import defaultdict

import numpy as np

from .utils.lpmln import Program, Rule, Literal


EPS = 1e-10          # Value used for numerical stabilization
U_W_PR = 0.999        # How much the agent values information provided by the user
SR_THRES = -math.log(0.5)     # Surprisal threshold


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
    models_v = prog.solve()
    marginals_v = models_v.marginals()

    return models_v, marginals_v, prog


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
        Any mismatches between vis-only vs. vis-and-lang marginals
    """
    models_v, marginals_v, prog = vis_result

    # Incorporate info from dialogue state
    aprog = Program()      # For finding referent-entity assignment
    dprog = Program()      # For incorporating additional info from user

    # Collect grounded facts and constraints provided by user
    facts = []; constraints = []
    for speaker, utt_type, content in dialogue_state["record"]:
        if speaker == "U" and utt_type == "|":
            for head, body, choice in content:
                # Grounded facts
                if body == None and not choice:
                    facts.append(head)
                
                # Grounded constraints
                if head == None:
                    constraints.append(body)
    
    # Find the best estimate of assignment

    if len(dialogue_state["referents"]["dis"]) == len(dialogue_state["assignment_hard"]):
        # Can take shortcut if all referents are hard-assigned to some entity
        best_assignment = dialogue_state["assignment_hard"]

    else:
        # Environment entities
        for atom, conf in marginals_v.items():
            pred = atom.name
            args = atom.args

            if pred == "object":
                aprog.add_hard_rule(Rule(head=Literal("env", args)))
            else:
                aprog.add_hard_rule(
                    Rule(head=Literal("env", args+[(pred, False), (int(100*float(conf)), False)]))
                )

        # Discourse referents
        for speaker, utt_type, content in dialogue_state["record"]:

            # Consider grounded statements from the user
            for f in facts:
                vis_concept = lexicon.s2d[(f[0], f[1])]
                args = [(a, False) for a in f[2]]
                args = args + [(f"{vis_concept[1]}_{vis_concept[0]}", False), (100, False)]
                aprog.add_hard_rule(Rule(head=Literal("dis", args)))

            for c in constraints:
                if len(c) == 1:
                    # Not a lot of hassle here, one line of fact will do
                    vis_concept = lexicon.s2d[(c[0][0], c[0][1])]
                    args = [(a, False) for a in c[0][2]]
                    args = args + [(f"{vis_concept[1]}_{vis_concept[0]}", False), (-50, False)]
                    aprog.add_hard_rule(Rule(head=Literal("dis", args)))
                else:
                    # Compose auxiliary predicates and rules for this specific conjunction.
                    # I know this is too much. Blame my OCD...
                    vis_concepts = [
                        lexicon.s2d[(bl[0], bl[1])] for bl in c
                    ]

                    var_inds = defaultdict(lambda: f"E{len(var_inds)}")

                    aux_pred = "&".join([f"{vc[1]}_{vc[0]}" for vc in vis_concepts])

                    aux_rule_body = [
                        Literal(
                            "env",
                            var_ls(*[var_inds[a] for a in bl[2]]) + \
                                [(f"{vc[1]}_{vc[0]}", False), (f"W{i}", True)]
                        )
                        for i, (bl, vc) in enumerate(zip(c, vis_concepts))
                    ]
                    aux_rule_head = Literal(
                        "env",
                        var_ls(*[vi for vi in var_inds.values()]) + \
                            [(aux_pred, False)] + \
                            [([op for i in range(len(c)) for op in (f"W{str(i)}", "*")][:-1], True)]
                    )

                    aprog.add_hard_rule(Rule(head=aux_rule_head, body=aux_rule_body))

                    args = [(a, False) for a in var_inds.keys()]
                    args = args + [(aux_pred, False), (-1.0, False)]
                    aprog.add_hard_rule(Rule(head=Literal("dis", args)))

        # Hard assignments by pointing, etc.
        for ref, env in dialogue_state["assignment_hard"].items():
            aprog.add_hard_rule(
                Rule(body=[Literal("assign", [(ref, False), (env, False)], naf=True)])
            )

        ## Assignment program rules
        # :- env(E,P,W), not env(E).
        aprog.add_hard_rule(Rule(
            body=[
                Literal("env", var_ls("E", "P", "W")),
                Literal("env", var_ls("E"), naf=True),
            ]))
        # :- env(E1,E2,P,W), not env(E1).
        aprog.add_hard_rule(Rule(
            body=[
                Literal("env", var_ls("E1", "E2", "P", "W")),
                Literal("env", var_ls("E1"), naf=True),
            ]
        ))
        # :- env(E1,E2,P,W), not env(E2).
        aprog.add_hard_rule(Rule(
            body=[
                Literal("env", var_ls("E1", "E2", "P", "W")),
                Literal("env", var_ls("E2"), naf=True),
            ]
        ))

        # dis(X) :- dis(X,P,S).
        aprog.add_hard_rule(Rule(
            head=Literal("dis", var_ls("X")),
            body=[
                Literal("dis", var_ls("X", "P", "S"))
            ]
        ))
        # dis(X1) :- dis(X1,X2,P,S).
        aprog.add_hard_rule(Rule(
            head=Literal("dis", var_ls("X1")),
            body=[
                Literal("dis", var_ls("X1", "X2", "P", "S"))
            ]
        ))
        # dis(X2) :- dis(X1,X2,P,S).
        aprog.add_hard_rule(Rule(
            head=Literal("dis", var_ls("X2")),
            body=[
                Literal("dis", var_ls("X1", "X2", "P", "S"))
            ]
        ))

        # { assign(X,E) : env(E) } 1 :- dis(X).
        aprog.add_rule(Rule(
            head=Literal(
                "assign", var_ls("X", "E"), conds=[Literal("env", var_ls("E"))]
            ),
            body=[Literal("dis", var_ls("X"))],
            ub=1
        ))

        # cons(X,E,P,W*S) :- dis(X,P,S), env(E,P,W), assign(X,E).
        aprog.add_hard_rule(Rule(
            head=Literal("cons", var_ls("X", "E", "P", ["W", "*", "S"])),
            body=[
                Literal("dis", var_ls("X", "P", "S")),
                Literal("env", var_ls("E", "P", "W")),
                Literal("assign", var_ls("X", "E")),
            ]
        ))
        # cons(X1,X2,E1,E2,P,W*S) :- dis(X1,X2,P,S), env(E1,E2,P,W), assign(X1,E1), assign(X2,E2).
        aprog.add_hard_rule(Rule(
            head=Literal("cons", var_ls("X1", "X2", "E1", "E2", "P", ["W", "*", "S"])),
            body=[
                Literal("dis", var_ls("X1", "X2," "P", "S")),
                Literal("env", var_ls("E1", "E2," "P", "W")),
                Literal("assign", var_ls("X1", "E1")),
                Literal("assign", var_ls("X2", "E2")),
            ]
        ))

        # By querying for the optimal assignment, essentially we are giving the user a 'benefit
        # of doubt', such that any statements made by the user are considered as True, and the
        # agent will try to find the 'best' assignment to make it so.
        # (Note: this is not a probabilistic inference, and the confidence scores provided as 
        # arguments are better understood as properties of the env. entities & disc. referents.)
        opt_models = aprog.optimize([
            ("maximize", [
                ([Literal("cons", var_ls("X", "E", "P", "W"))], "W", ["X", "E", "P"]),
                ([Literal("cons", var_ls("X1", "X2", "E1", "E2", "P", "W"))], "W", ["X1", "X2", "E1", "E2", "P"])
            ])
        ])

        best_assignment = [atom.args for atom in opt_models[0] if atom.name == "assign"]
        best_assignment = {str(args[0][0]): str(args[1][0]) for args in best_assignment}

    a_map = lambda args: [best_assignment[a] for a in args]

    # Now incorporate additional information provided by the user in language for updated
    # sensemaking
    for f in facts:
        vis_concept = lexicon.s2d[(f[0], f[1])]
        pred = f"{vis_concept[1]}_{vis_concept[0]}"
        args = [(a, False) for a in a_map(f[2])]

        dprog.add_rule(Rule(head=Literal(pred, args)), U_W_PR)

    for c in constraints:
        vis_concepts = [lexicon.s2d[(bl[0], bl[1])] for bl in c]
        constr_body = [
            Literal(f"{vc[1]}_{vc[0]}", [(a, False) for a in a_map(bl[2])])
            for bl, vc in zip(c, vis_concepts)
        ]

        dprog.add_rule(Rule(body=constr_body), U_W_PR)
    
    # Finally, reasoning with all visual+language info
    prog += dprog
    models_vl = prog.solve()
    marginals_vl = models_vl.marginals()

    # Identify any drastic mismatches between vis only vs. lang input
    mismatches = []

    # Test grounded facts
    for f in facts:
        vis_concept = lexicon.s2d[(f[0], f[1])]
        f_vis = (vis_concept[0], vis_concept[1], a_map(f[2]))

        ev_prob = models_v.query(
            Literal(f"{f_vis[1]}_{f_vis[0]}", [(a, False) for a in f_vis[2]])
        )

        surprisal = -math.log(ev_prob + EPS)
        if surprisal > SR_THRES:
            mismatches.append((True, f, f_vis, surprisal))

    # Test grounded constraints
    for c in constraints:
        vis_concepts = [lexicon.s2d[(bl[0], bl[1])] for bl in c]
        c_vis = [
            (vc[0], vc[1], a_map(bl[2])) for bl, vc in zip(c, vis_concepts)
        ]
        c_atoms = [
            Literal(f"{vc1}_{vc0}", [(a, False) for a in ents])
            for vc0, vc1, ents in c_vis
        ]

        ev_prob = models_v.query(c_atoms, neg=True)

        surprisal = -math.log(ev_prob + EPS)
        if surprisal > SR_THRES:
            mismatches.append((False, c, c_vis, surprisal))

    return models_vl, marginals_vl, prog, mismatches


def var_ls(*vs):
    return [(v, True) for v in vs]
