"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import math
from collections import defaultdict

import clingo


def top_k_models(prog, k):
    """Find the top k models with the highest total weights for the given program"""
    top_k = []
    best = None
    ctl = None

    while len(top_k) < k:
        print(f"Let me see... ({len(top_k)}/{k})" , end="\r")

        if ctl is None:
            ctl = clingo.Control()
            ctl.add("base", [], prog)
            ctl.ground([("base", [])])
            ctl.configuration.solve.models = 0
            ctl.configuration.solve.opt_mode = "optN"

        if best is not None:
            wcprog = f":- {best} >= #sum {{ W,R : unsat(R, W) }}."
            ctl.add("wcons", [], wcprog)
            ctl.ground([("wcons", [])])

        models = []

        with ctl.solve(yield_=True) as solve_gen:

            if solve_gen.get().unsatisfiable: break

            for m in solve_gen:
                models.append((m.symbols(atoms=True), m.cost[0], m.optimality_proven))
                if m.optimality_proven:
                    best = m.cost[0]

        models = [m for m in models if m[1] == best]
        models = {",".join([str(a) for a in m[0] if a.name != "unsat"]): (m[0], m[1]) for m in models}

        top_k += list(models.values())

    return top_k


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
