""" Program().reduce() factored out """
import copy

from ..utils import logit


def reduce_program(prog, pos_atoms, neg_atoms):
    """ Return the program obtained by reducing prog with given values of atoms """
    from .program import Program

    reduced_rules = copy.deepcopy(prog.rules)
    rules_to_del = set()

    # Sum of weights harvested by excluding rules whose bodies are rendered
    # unsatisfiable with this reduction
    base_weight = [0, 0]        # (zeroth degree of "a", first degree of "a")

    for pa in pos_atoms:
        for ri in prog._rules_by_atom[pa.as_atom()]:
            rule = reduced_rules[ri][0]
            if rule.body_contains(pa.flip()):
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
            if pa in rule.body: rule.body.remove(pa)

    for na in neg_atoms:
        for ri in prog._rules_by_atom[na.as_atom()]:
            rule = reduced_rules[ri][0]
            if rule.body_contains(na):
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

            # Remove negative appearance if exists
            if na.flip() in rule.body: rule.body.remove(na.flip())

    reduced_rules = [
        r for i, r in enumerate(reduced_rules) if i not in rules_to_del
    ]

    return Program(reduced_rules), base_weight
