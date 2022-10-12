""" Models().filter() factored out """
import operator
from itertools import product
from functools import reduce

from ..literal import Literal
from ..polynomial import Polynomial


def filter_models(models, literals):
    """
    Filter the Models instance to return a new Models instance representing the
    subset of models satisfying the condition specified by the *disjunction* of
    literals, so that formulas in conjunctive normal form can be processed with
    a chain of calls to filter.
    """
    if models.is_empty():
        # Empty Models instance
        return models

    if type(literals) != frozenset:
        try:
            # Treat as set
            literals = frozenset(literals)
        except TypeError:
            # Accept single-literal disjunction and wrap in a set
            assert isinstance(literals, Literal)
            literals = frozenset([literals])

    if hasattr(models, "factors"):
        return _filter_factors(models, literals)

    if hasattr(models, "outcomes"):
        return _filter_outcomes(models, literals)

    raise ValueError("Invalid Models instance")

def _filter_factors(models, literals):
    """
    Factorizing out submethod to be called by filter_models(), for factors-Models
    instances
    """
    assert hasattr(models, "factors")

    from .models import Models

    # Sets of atoms actually covered by the factors
    f_atoms = [
        f.atoms() if isinstance(f, Models) else
            ({f[0]} if f[2] is None or f[2]==True else set())
        for f in models.factors
    ]

    # Relevant factor for each literal in disjunction
    lit_relevant_factors = {
        l: {i for i, atms in enumerate(f_atoms) if l.as_atom() in atms}
        for l in literals
    }
    # (Invariant: Each literal should have at most one relevant factor)
    assert all([len(s)<=1 for s in lit_relevant_factors.values()])

    for l in literals:
        if len(lit_relevant_factors[l])==0:
            if l.naf:
                ## l.naf==True; l is negative literal
                # If a negative literal in disjunction is not covered by any
                # factor, the disjunction is trivially satisfiable and models
                # can be returned right away
                return models
            else:
                ## l.naf==False; l is positive literal
                # If a positive literal in disjunction is not covered by any
                # factor, the literal is not satisfiable by any model this Models
                # instance contains, and thus can be removed from the provided
                # disjunction
                lit_relevant_factors.pop(l)
    
    if len(lit_relevant_factors)==0:
        # Disjunction not by satisfied by any model; return empty Models instance
        return Models()

    # Factors are either kept intact or filtered with respect to relevant
    # literal(s), then combined to yield a new list of factors
    filtered_factors = []
    intact_factors = []
    for i, f in enumerate(models.factors):
        covered_lits = {
            l for l, fi in lit_relevant_factors.items() if i in fi
        }

        if len(covered_lits) == 0:
            # Factor irrelevant to any literal; include as-is
            intact_factors.append(f)
        else:
            # Factor needs filtering by the covered literals
            if isinstance(f, Models):
                # Recurse; another Models instance
                filtered_factors.append((filter_models(f, covered_lits), i))
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3
                if f[0] is None:
                    # Signals violation of integrity constraint; disregard
                    continue

                assert isinstance(f[0], Literal)
                assert len(covered_lits) == 1
                lit = covered_lits.pop()
                collapse_val = not lit.naf     # Recall naf==True means negated literal

                # Collapse into matching truth value; note value of f[2] should
                # be either None or True here
                filtered_factors.append(((f[0], f[1], collapse_val), i))
    
    if len(filtered_factors) == 1:
        # Simpler case with only one filtered factor
        filtered_factors = [f for f, _ in filtered_factors]

    else:
        # If we have more than one filtered factors, representation of the exact set of
        # models satisfying the disjunction cannot be achieved with independent factors
        # and these should be merged into a single outcome-Models instance.
        # This will naturally lead to exponential increase in size of data needed to
        # describe the data structure, meaning longer disjunctions are likely to result
        # in less compact representation of model sets: more enumeration, less factoring.
        # (Not that it can be avoided -- it's still the best we can do!)
        assert len(filtered_factors) > 1

        # Method that returns (frozen)set of models covered by a factor enumerated
        def enum_f(f):
            if isinstance(f, Models):
                return set(f.enumerate())
            else:
                f_lit, f_w, f_coll = f
                if f_w is not None:
                    # Non-absolute incidental rule, hard or soft
                    f_w_poly = Polynomial.from_primitive(f_w)
                    lit_p = (frozenset([f_lit]), f_w_poly)
                    lit_n = (frozenset(), Polynomial(float_val=1.0))

                    if f_coll is None:
                        return {lit_p, lit_n}
                    else:
                        if f_coll:
                            return {lit_p}
                        else:
                            return {lit_n}
                else:
                    # Absolute rule that doesn't allow failure to derive f_lit
                    lit_p = (frozenset([f_lit]), Polynomial(float_val=1.0))

                    if f_coll is None:
                        return {lit_p}
                    else:
                        if f_coll:
                            return {lit_p}
                        else:
                            return set()

        # Models enumerated for each filtered factor, and their complements with respect
        # to the model enumeration for the original factor before filtering. Required for
        # computing the final enumeration satisfying the disjunction.
        f_enum_models = [(enum_f(f), i) for f, i in filtered_factors]           
        f_enum_models = [
            (ms, enum_f(models.factors[i])-ms) for ms, i in f_enum_models
        ]

        # Enumeration of all possible models from which set of models not satisfying the
        # disjunction will be subtracted
        all_enum_models = set.union(*[
            {
                (
                    frozenset.union(*[model for model, _ in model_choices]),
                    reduce(operator.mul, [ws for _, ws in model_choices])
                )
                for model_choices in product(*bin_choices)
            }
            for bin_choices in product(*f_enum_models)
        ])
        
        # Enumeration of models that fail to satisfy the disjunction
        unsat_enum_models = {
            (
                frozenset.union(*[model for model, _ in model_choices]),
                reduce(operator.mul, [ws for _, ws in model_choices])
            )
            for model_choices
            in product(*[filtered_comp for _, filtered_comp in f_enum_models])
        }

        bottom_models_merged = Models(factors=[
            models.factors[i] for _, i in filtered_factors
        ])
        possible_atoms = bottom_models_merged.atoms()

        cases_merged = [
            (case | {na.flip() for na in possible_atoms-case}, ws)
            for case, ws in all_enum_models - unsat_enum_models
        ]       # Making negative atoms explicit
        outcomes_merged = (
            bottom_models_merged,
            [
                (case, ws, Polynomial(float_val=1.0), Models(factors=[]), False)
                for case, ws in cases_merged
            ]
        )

        filtered_factors = [Models(outcomes=outcomes_merged)]

    return Models(factors=filtered_factors+intact_factors)

def _filter_outcomes(models, literals):
    """
    Factorizing out submethod to be called by filter_models(), for outcomes-Models
    instances
    """
    assert hasattr(models, "outcomes")

    from .models import Models

    bottom_models, branch_consequences = models.outcomes

    if branch_consequences is None:
        return Models(outcomes=(filter_models(bottom_models, literals), None))
    else:
        # Filter each outcome and add to new outcome list if result is not empty
        filtered_consequences = []

        for bc in branch_consequences:
            # Each outcome entry is a tuple of a model for some program bottom in
            # the splitting sequence of the program (tree actually), a weight sum
            # applied to the bottom model, a Models instance for some program top,
            # and a boolean flag denoting whether this outcome branch is filtered
            # out and thus shouldn't contribute to 'covered' probability mass
            branch_ev, bm_filtered, branch_w, top_base_w, top_models, filtered_out = bc

            if filtered_out:
                # This outcome branch is already filtered out, can add as-is right
                # away
                filtered_consequences.append(bc)
                continue

            # Set of atoms actually covered by the outcome
            branch_atoms = bm_filtered.atoms()
            top_atoms = top_models.atoms()
            bc_atoms = branch_atoms | top_atoms

            # Disjunction to satisfy; making a copy that we can manipulate within this
            # for loop
            disjunction = {l for l in literals}

            add_as_is = False
            for l in literals:
                # literal shouldn't be covered by both branch_atoms and top_atoms
                assert not ((l.as_atom() in branch_atoms) and (l.as_atom() in top_atoms))

                if l.naf:
                    ## l.naf==True; l is negative literal
                    # If not covered by either branch_atoms or top_atoms, the disjunction
                    # is trivially satisfiable and this outcome can be included as-is without
                    # filtering
                    if l.as_atom() not in bc_atoms:
                        add_as_is = True

                    # If covered directly by branch_ev, this literal is never satisfiable and
                    # can be removed from disjunction
                    if l.as_atom() in branch_ev:
                        disjunction.remove(l)

                else:
                    ## l.naf==False; l is positive literal
                    # If covered directly by branch_ev, the disjunction is trivially satisfiable
                    # and this outcome can be included as-is without filtering
                    if l in branch_ev:
                        add_as_is = True

                    # If not covered by either branch_atoms or top_atoms, this literal is
                    # never satisfiable and can be removed from disjunction
                    if l not in bc_atoms:
                        disjunction.remove(l)

            if add_as_is:
                # No filtering needed, add the outcome as-is
                filtered_consequences.append(bc)
                continue
            if len(disjunction) == 0:
                # Empty disjunction cannot be satisfied; flag as filtered and add to
                # outcome list
                filtered_consequences.append(
                    (branch_ev, bm_filtered, branch_w, top_base_w, top_models, True)
                )
                continue

            # Outcomes that reached here represent mixture of models satisfying the
            # disjunction and those that do not; need further filtering of bm_filtered
            # and top_models, depending on how literals are covered by each
            literals_bm_f = {l for l in literals if l.as_atom() in branch_atoms}
            literals_top = {l for l in literals if l.as_atom() in top_atoms}

            if len(literals_bm_f) > 0:
                bm_further_filtered = filter_models(bm_filtered, literals_bm_f)
                branch_flt_w = bm_further_filtered.compute_Z()
            else:
                bm_further_filtered = bm_filtered
                branch_flt_w = branch_w

            if len(literals_top) > 0:
                top_models_filtered = filter_models(top_models, literals_top)
            else:
                top_models_filtered = top_models

            filtered_consequences.append(
                (
                    branch_ev, bm_further_filtered, branch_flt_w,
                    top_base_w, top_models_filtered, filtered_out
                )
            )

        return Models(outcomes=(bottom_models, filtered_consequences))
