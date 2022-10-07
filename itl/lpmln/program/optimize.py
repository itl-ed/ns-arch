""" Program().optimize() factored out """
import clingo

from ..literal import Literal


def optimize(prog, statements):
    """
    Solve for the optimal model as specified by [statements] argument. Optimal solution
    is found by composing appropriate optimization statements in clingo, which is attached
    to string representation of the program, and solving the program with clingo.
    
    Each optimization statement designates how much weight each occurrence of relevant
    literals (counted at most once for tuple) should contribute to the total weight.
    Statements that come earlier in the list will have higher priority.
    """
    stm_asp_str = ""
    for p, stm in enumerate(statements[::-1]):
        # Each statement may consist of more than one weight formulas: that is, (literals,
        # weight, terms) tuples; weights will be summed in the priority level
        max_or_min, formulas = stm
        assert max_or_min == "maximize" or max_or_min == "minimize", \
            "Can only 'maximize' or 'minimize'"

        formulas_asp_str = [
            f"{weight}@{p}{','+','.join(terms) if len(terms)>0 else ''} : "
                f"{','.join([str(l) for l in lits]) if len(lits)>0 else ''}"
            for lits, weight, terms in formulas
        ]

        stm_asp_str += f"#{max_or_min} {{ {'; '.join(formulas_asp_str)} }}.\n"

    # Optimize with clingo
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], prog._pure_ASP_str()+stm_asp_str)
    ctl.ground([("base", [])])
    ctl.configuration.solve.models = 0
    ctl.configuration.solve.opt_mode = "opt"

    models = []; best_cost = [float("inf")] * len(statements)
    with ctl.solve(yield_=True) as solve_gen:
        for m in solve_gen:
            models.append((m.symbols(atoms=True), m.cost))
            if m.cost[::-1] < best_cost[::-1]: best_cost = m.cost
            if solve_gen.get().unsatisfiable: break
    
    models = [m[0] for m in models if m[1] == best_cost]
    models = [
        [Literal.from_clingo_symbol(a) for a in m] for m in models
    ]

    return models
