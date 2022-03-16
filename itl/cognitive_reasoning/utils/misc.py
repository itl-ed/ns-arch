"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import math
from collections import defaultdict

import clingo


def sample_from_top(prog, k, s=10, w=100):
    """
    Find the best model(s) with the highest total weights for the given program, then
    keep sampling models within a slinding cost window until k models are sampled.
    """
    sampled = []

    # First find the best model and its total cost (weight sum)
    opt_prog = f"#minimize {{ W@0,R : unsat(R,W) }}.\n"
    ctl = clingo.Control()
    ctl.add("base", [], prog+opt_prog)
    ctl.ground([("base", [])])
    ctl.configuration.solve.models = 0
    ctl.configuration.solve.opt_mode = "optN"

    models = []
    with ctl.solve(yield_=True) as solve_gen:
        for m in solve_gen:
            models.append((m.symbols(atoms=True), m.cost[0]))
            if m.optimality_proven:
                cost_floor = m.cost[0]
    
    # models = {
    #     ",".join([str(a) for a in m[0] if a.name != "unsat"]): (m[0], m[1])
    #     for m in models
    # }
    sampled += [m for m in models if m[1]==cost_floor]

    # Then keep sampling models within a sliding cost window until we have k models
    while len(sampled) < k:
        print(f"Let me see... ({len(sampled)}/{k})" , end="\r")

        ctl = clingo.Control()
        ctl.add("base", [], prog)
        ctl.ground([("base", [])])
        ctl.configuration.solve.models = s

        wcprog = \
            f":- {cost_floor} >= #sum {{ W,R : unsat(R, W) }}.\n" \
            f":- {cost_floor+w} < #sum {{ W,R : unsat(R, W) }}.\n"
        ctl.add("wcons", [], wcprog)
        ctl.ground([("wcons", [])])

        models = []
        with ctl.solve(yield_=True) as solve_gen:
            for m in solve_gen:
                models.append(m.symbols(atoms=True))

        models = [
            (m, sum([a.arguments[1].number for a in m if a.name=="unsat"]))
            for m in models
        ]
        sampled += models

        cost_floor += w

    return sampled


def marginalize(models):
    """Compute marginal probabilities for literals from the given set of models"""
    Z = 0  # Partition function
    marginals = defaultdict(lambda: 0)

    for m in models:
        Z += math.exp(m[1])
        for atom in m[0]:
            if atom.name == "unsat": continue

            atom = (atom.name, tuple([arg.name for arg in atom.arguments]))
            marginals[atom] += math.exp(m[1])

    marginals = dict({k: v/Z for k, v in marginals.items()})

    return marginals
