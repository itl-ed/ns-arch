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

from .utils import sample_from_top, marginalize


SCALE_PREC = 3e2   # For preserving some float weight precision
LARGE = 2e1        # Sufficiently large logit to use in place of, say, float('inf')
EPS = 1e-10        # Value used for numerical stabilization

K_M = 300          # Number of models to sample, from the most optimal

U_WEIGHT = 8       # How much the agent values information provided by the user

SR_THRES = -math.log(0.5)     # Surprisal threshold


def sensemake_vis(vis_scene, objectness_thresh=0.3, category_thresh=0.2):
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
    prog = ""

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
    pprog = ""
    rule_count = 0
    classes_all = set(); attributes_all = set(); relations_all = set()

    for oi, obj in vis_scene.items():
        # Objectness
        atom = f"object({oi})"
        weight = _logit(obj["pred_objectness"])
        weight = int(weight * SCALE_PREC)

        pprog += f"{{ {atom} }}.\n"
        pprog += f"unsat({rule_count},{weight}) :- not {atom}.\n"
        rule_count += 1

        # Object classes
        classes = set(np.where(obj["pred_classes"] > category_thresh)[0])
        classes.add(obj["pred_classes"].argmax())    # Also add the max category
        classes_all |= classes
        for c in classes:
            atom = f"cls_{c}({oi})"
            weight = _logit(obj["pred_classes"][c])
            weight = int(weight * SCALE_PREC)

            pprog += f"{{ {atom} }}.\n"
            pprog += f"unsat({rule_count},{weight}) :- not {atom}.\n"
            rule_count += 1

        # Object attributes
        attributes = set(np.where(obj["pred_attributes"] > category_thresh)[0])
        attributes.add(obj["pred_attributes"].argmax())
        attributes_all |= attributes
        for a in attributes:
            atom = f"att_{a}({oi})"
            weight = _logit(obj["pred_attributes"][a])
            weight = int(weight * SCALE_PREC)

            pprog += f"{{ {atom} }}.\n"
            pprog += f"unsat({rule_count},{weight}) :- not {atom}.\n"
            rule_count += 1
        
        # Object relations
        relations = {
            oj: set(np.where(per_obj > category_thresh)[0])
            for oj, per_obj in obj["pred_relations"].items()
        }
        for oj, per_obj in relations.items():
            per_obj.add(obj["pred_relations"][oj].argmax())
            relations_all |= per_obj
            for r in per_obj:
                atom = f"rel_{r}({oi},{oj})"
                weight = _logit(obj["pred_relations"][oj][r])
                weight = int(weight * SCALE_PREC)

                pprog += f"{{ {atom} }}.\n"
                pprog += f"unsat({rule_count},{weight}) :- not {atom}.\n"
                rule_count += 1

    # Constraints precluding predicates without objectness
    classes_cnst = "\n ".join([
        f":- cls_{c}(O), not object(O)." for c in classes_all
    ])
    attributes_cnst = "\n ".join([
        f":- att_{a}(O), not object(O)." for a in attributes_all
    ])
    relations_cnst = "\n ".join([
        f":- rel_{r}(O1,O2), not object(O1). :- rel_{r}(O1,O2), not object(O2)."
        for r in relations_all
    ])

    pprog += \
        f"{classes_cnst}\n " \
        f"{attributes_cnst}\n " \
        f"{relations_cnst}\n "

    # Solve with clingo to find the best K_M models of the program
    prog += pprog
    models_v = sample_from_top(prog, K_M)

    # Rescale weights, and don't forget to flip signs of the penalties
    models_v = [(m[0], -m[1]/SCALE_PREC) for m in models_v]

    # Subtract minimum weight sum for numerical stability
    min_ws = min([m[1] for m in models_v])
    models_v = [(m[0], m[1]-min_ws) for m in models_v]

    # Compute marginals
    marginals_v = marginalize(models_v)

    return models_v, marginals_v, prog, rule_count


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
    models_v, marginals_v, prog, rule_count = vis_result

    # Incorporate info from dialogue state
    aprog = ""  # For finding referent-entity assignment
    dprog = ""  # For incorporating additional info from user

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
            pred, args = atom

            if pred == "object":
                aprog += f"env({args[0]}).\n"
            else:
                weight = _logit(conf)
                weight = int(weight * SCALE_PREC)
                aprog += f"env({','.join(args)},\"{pred}\",{weight}).\n"

        # Discourse referents
        for speaker, utt_type, content in dialogue_state["record"]:

            # Consider grounded statements from the user
            for f in facts:
                vis_concept = lexicon.s2d[(f[0], f[1])]
                aprog += f"dis({','.join(f[2])},\"{vis_concept[1]}_{vis_concept[0]}\",1).\n"
            
            for c in constraints:
                if len(c) == 1:
                    # Not a lot of hassle here, one line of fact will do
                    vis_concept = lexicon.s2d[(c[0][0], c[0][1])]
                    aprog += f"dis({','.join(c[0][2])},\"{vis_concept[1]}_{vis_concept[0]}\",-1).\n"
                else:
                    # Compose auxiliary predicates and rules for this specific conjunction.
                    # I know this is too much. Blame my OCD...
                    vis_concepts = [
                        lexicon.s2d[(bl[0], bl[1])] for bl in c
                    ]

                    var_inds = defaultdict(lambda: f"E{len(var_inds)}")

                    aux_pred = "&".join([f"{vc[1]}_{vc[0]}" for vc in vis_concepts])

                    aux_rule_body = [
                        f"env({','.join([var_inds[a] for a in bl[2]])},\"{vc[1]}_{vc[0]}\",W{i})"
                        for i, (bl, vc) in enumerate(zip(c, vis_concepts))
                    ]
                    aux_rule_body = ", ".join(aux_rule_body)

                    aux_head_args = ','.join(var_inds.values())
                    aux_weight_prods = "*".join([f"W{str(i)}" for i in range(len(c))])
                    aux_weight_prods += f"/{int(SCALE_PREC)}" * (len(c) - 1)
                    aux_rule_head = f"env({aux_head_args},\"{aux_pred}\",{aux_weight_prods})"

                    aprog += f"{aux_rule_head} :- {aux_rule_body}.\n"
                    aprog += f"dis({','.join(var_inds.keys())},\"{aux_pred}\",-1).\n"

        # Hard assignments by pointing, etc.
        for ref, env in dialogue_state["assignment_hard"].items():
            aprog += f":- not assign({ref},{env}).\n"

        aprog += \
            ":- env(E,P,W), not env(E). :- env(E1,E2,P,W), not env(E1). :- env(E1,E2,P,W), not env(E2).\n" \
            "dis(X) :- dis(X,P,S). dis(X1) :- dis(X1,X2,P,S). dis(X2) :- dis(X1,X2,P,S).\n" \
            "{ assign(X,E) : env(E) } 1 :- dis(X).\n" \
            "cons(X,E,P,W*S) :- dis(X,P,S), env(E,P,W), assign(X,E).\n" \
            "cons(X1,X2,E1,E2,P,W*S) :- dis(X1,X2,P,S), env(E1,E2,P,W), assign(X1,E1), assign(X2,E2).\n" \
            f"#maximize {{ W@0,X,E,P : cons(X,E,P,W); W@0,X1,X2,E1,E2,P : cons(X1,X2,E1,E2,P,W) }}.\n"

        # By querying for the optimal assignment, essentially we are giving the user a 'benefit
        # of doubt', such that any statements made by the user are considered as True, and the
        # agent will try to find the 'best' assignment to make it so.
        models_a = sample_from_top(aprog, 1)

        best_assignment = [atom.arguments for atom in models_a[0][0] if atom.name == "assign"]
        best_assignment = {str(args[0]): str(args[1]) for args in best_assignment}

    a_map = lambda args: [best_assignment[a] for a in args]

    # Now incorporate additional information provided by the user in language for updated
    # sensemaking
    weight = int(U_WEIGHT*SCALE_PREC)

    for f in facts:
        vis_concept = lexicon.s2d[(f[0], f[1])]
        atom = f"{vis_concept[1]}_{vis_concept[0]}({','.join(a_map(f[2]))})"
        dprog += f"{{ {atom} }}.\n"
        dprog += f"unsat({rule_count},{weight}) :- not {atom}.\n"
        rule_count += 1

    for c in constraints:
        vis_concepts = [lexicon.s2d[(bl[0], bl[1])] for bl in c]
        constr_body = [
            f"{vc[1]}_{vc[0]}({','.join(a_map(bl[2]))})"
            for bl, vc in zip(c, vis_concepts)
        ]
        dprog += f"unsat({rule_count},{weight}) :- {', '.join(constr_body)}.\n"
        rule_count += 1
    
    # Finally, reasoning with all visual+language info
    prog += dprog
    models_vl = sample_from_top(prog, K_M)
    models_vl = [(m[0], -m[1]/SCALE_PREC) for m in models_vl]

    min_ws = min([m[1] for m in models_vl])
    models_vl = [(m[0], m[1]-min_ws) for m in models_vl]
    marginals_vl = marginalize(models_vl)

    ## Identify any drastic mismatches between vis only vs. lang input

    mismatches = []

    Z = sum([math.exp(m[1]) for m in models_v])
    models_v_set = [
        ({str(a) for a in m[0] if a.name != "unsat"}, m[1])
        for m in models_v
    ]

    # Test grounded facts
    for f in facts:
        vis_concept = lexicon.s2d[(f[0], f[1])]
        f_vis = (vis_concept[0], vis_concept[1], a_map(f[2]))
        f_serialised = f"{f_vis[1]}_{f_vis[0]}({','.join(f_vis[2])})"

        true_sum = sum([math.exp(m[1]) for m in models_v_set if f_serialised in m[0]])

        surprisal = -math.log((true_sum+EPS) / Z)
        if surprisal > SR_THRES:
            mismatches.append((True, f, f_vis, surprisal))

    # Test grounded constraints
    for c in constraints:
        vis_concepts = [lexicon.s2d[(bl[0], bl[1])] for bl in c]
        c_vis = [
            (vc[0], vc[1], a_map(bl[2])) for bl, vc in zip(c, vis_concepts)
        ]
        c_serialised = [
            f"{vc1}_{vc0}({','.join(ents)})" for vc0, vc1, ents in c_vis
        ]
        test_constr = lambda m: not all([(bl in m) for bl in c_serialised])

        true_sum = sum([math.exp(m[1]) for m in models_v_set if test_constr(m[0])])

        surprisal = -math.log((true_sum+EPS) / Z)
        if surprisal > SR_THRES:
            mismatches.append((False, c, c_vis, surprisal))

    return models_vl, marginals_vl, prog, mismatches


def _logit(p):
    """Compute logit of the probability value p, capped by LARGE value (+/-)"""
    if p == 1:
        return LARGE
    elif p == -1:
        return -LARGE
    else:
        return math.log(p/(1-p))
