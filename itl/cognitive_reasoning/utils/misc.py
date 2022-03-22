"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import math
from collections import defaultdict

import clingo

from .xorsample import xorsample


def sample_models(prog, k, s=1, w=5000):
    """
    Sample k models within a sliding cost window moving from the best model cost to
    worse, resetting each time it hits the bottom (i.e. unsatisfiable)
    """
    # First find the best model and its total cost (weight sum)
    # opt_prog = f"#minimize {{ W@0,R : unsat(R,W) }}.\n"
    # ctl = clingo.Control()
    # ctl.add("base", [], prog+opt_prog)
    # ctl.ground([("base", [])])
    # ctl.configuration.solve.models = 0
    # ctl.configuration.solve.opt_mode = "opt"

    sampled = []
    # cost_floor = float("inf")
    # with ctl.solve(yield_=True) as solve_gen:
    #     for m in solve_gen:
    #         print(f"Let me see... ({m.cost[0]})" , end="\r")
    #         sampled.append((m.symbols(atoms=True), m.cost[0]))
    #         if m.cost[0] < cost_floor:
    #             cost_floor = m.cost[0]

    # best_model = [m for m in sampled if m[1] == cost_floor]
    # sampled = best_model

    # Then keep sampling models within a sliding cost window until we have k models
    while len(sampled) < k:
        print(f"Let me see... ({len(sampled)}/{k})" , end="\r")

        # cost_ceil = cost_floor+w
        wcprog = ""
        # wcprog = \
        #     f":- {cost_floor} >= #sum {{ W,R : unsat(R, W) }}.\n" \
        #     f":- {cost_ceil} < #sum {{ W,R : unsat(R, W) }}.\n"

        models = xorsample(prog+wcprog, s)
        sampled += models

        # if len(models) > 0:
        #     cost_floor += w
        # else:
        #     cost_floor = best_model[0][1]

    return sampled


def marginalize(models):
    """Compute marginal probabilities for literals from the given set of models"""
    Z = 0  # Partition function
    marginals = defaultdict(lambda: 0)

    for m in models:
        Z += math.exp(m[1])

        non_unsats = [
            atom for atom in m[0]
            if atom.name != "unsat" and atom.name != "__parity"
        ]
        for atom in non_unsats:
            atom = (atom.name, tuple([arg.name for arg in atom.arguments]))
            marginals[atom] += math.exp(m[1])

    marginals = dict({k: v/Z for k, v in marginals.items()})

    return marginals
