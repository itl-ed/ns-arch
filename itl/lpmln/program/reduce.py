""" Program().reduce() factored out """
import copy

import numpy as np

from ..polynomial import Polynomial
from ..utils import logit


def reduce_program(prog, atoms):
    """ Return the program obtained by reducing prog with given values of atoms """
    from .program import Program

    reduced_rules = copy.deepcopy(prog.rules)
    rules_to_del = set()

    # Sum of weights harvested by excluding rules whose bodies are rendered
    # unsatisfiable with this reduction
    base_weight = [0, 0]        # (zeroth degree of "a", first degree of "a")

    # Positive atoms in the provided atom set are marked with naf==False, where
    # negative ones are marked with naf==True
    for atm in atoms:
        for ri in prog._rules_by_atom[atm.as_atom()]:
            rule = reduced_rules[ri][0]
            if rule.body_contains(atm.flip()):
                # Exclude this rule
                rules_to_del.add(ri)

                # Harvest weight of this rule as this will be necessarily
                # satisfied by this reduction
                r_pr = prog.rules[ri][1]
                if r_pr is None:
                    # Rule not contributing to weights, don't need to do
                    # a thing
                    pass
                else:
                    if len(r_pr) > 1:
                        # Don't know yet what has to happen here
                        raise NotImplementedError
                    else:
                        assert len(r_pr) == 1
                        weight = logit(r_pr[0], large="a")
                        if weight == "a":
                            base_weight[1] += 1
                        elif weight == "-a":
                            base_weight[1] -= 1
                        else:
                            base_weight[0] += weight

                continue

            # Remove positive appearance if exists
            if atm in rule.body: rule.body.remove(atm)

    reduced_rules = [
        r for i, r in enumerate(reduced_rules) if i not in rules_to_del
    ]
    base_weight_exp = Polynomial(terms={ base_weight[1]: base_weight[0] })

    return Program(reduced_rules), base_weight_exp
