""" Models().filter() factored out """
from itertools import product
from functools import reduce

from ..literal import Literal


def filter(models, literals):
    """
    Filter the Models instance to return a new Models instance representing the
    subset of models satisfying the condition specified by the *disjunction* of
    literals, so that formulas in conjunctive normal form can be processed with
    a chain of calls to filter.
    """
    if models.is_empty():
        # Empty Models instance
        return models

    if type(literals) != set:
        try:
            # Treat as set
            literals = set(literals)
        except TypeError:
            # Accept single-literal disjunction and wrap in a set
            assert isinstance(literals, Literal)
            literals = {literals}

    if hasattr(models, "factors"):
        return _filter_factors(models, literals)

    if hasattr(models, "outcomes"):
        return _filter_outcomes(models, literals)

    raise ValueError("Invalid Models instance")

def _filter_factors(models, literals):
    """
    Factorizing out submethod to be called by filter(), for factors-Models
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
                filtered_factors.append((filter(f, covered_lits), i))
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3 and isinstance(f[0], Literal)
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

        # This part has not been updated to comply with recent changes, and we don't know
        # what would happen here...
        raise NotImplementedError

        # Method that returns (frozen)set of models covered by a factor enumerated
        def enum_f(f):
            if isinstance(f, Models):
                return set(f.enumerate())
            else:
                lit_p = (frozenset([f[0]]), f[1])
                lit_n = (frozenset(), 1-f[1])

                if f[2] is None:
                    return {lit_p, lit_n}
                else:
                    if f[2]:
                        return {lit_p}
                    else:
                        return {lit_n}

        # Models enumerated for each filtered factor, and their complements with respect
        # to the model enumeration for the original factor before filtering. Required for
        # computing the final enumeration satisfying the disjunction.
        f_enum_models = [(enum_f(f), i) for f, i in filtered_factors]           
        f_enum_models = [
            (ms, enum_f(models.factors[i])-ms) for ms, i in f_enum_models
        ]

        # Enumeration all possible models from which set of models not satisfying the
        # disjunction will be subtracted
        all_enum_models = set.union(*[
            {
                (
                    frozenset.union(*[mc[0] for mc in model_choices]),
                    reduce(lambda p1,p2: p1*p2, [mc[1] for mc in model_choices])
                )
                for model_choices in product(*bin_choices)
            }
            for bin_choices in product(*f_enum_models)
        ])
        
        # Enumeration of models that fail to satisfy the disjunction
        unsat_enum_models = {
            (
                frozenset.union(*[mc[0] for mc in model_choices]),
                reduce(lambda p1,p2: p1*p2, [mc[1] for mc in model_choices])
            )
            for model_choices
            in product(*[filtered_comp for _, filtered_comp in f_enum_models])
        }

        merged_outcomes = all_enum_models - unsat_enum_models
        merged_outcomes = [(pr, set(model), None) for model, pr in merged_outcomes]

        filtered_factors = [Models(outcomes=merged_outcomes)]

    return Models(factors=filtered_factors+intact_factors)

def _filter_outcomes(models, literals):
    """
    Factorizing out submethod to be called by filter(), for outcomes-Models
    instances
    """
    assert hasattr(models, "outcomes")

    from .models import Models

    # Filter each outcome and add to new outcome list if result is not empty
    filtered_outcomes = []
    for o in models.outcomes:
        # Each outcome entry is a tuple of a model for some program bottom in
        # the splitting sequence of the program (tree actually), a weight sum
        # applied to the bottom model, a Models instance for some program top,
        # and a boolean flag denoting whether this outcome branch is filtered
        # out and thus shouldn't contribute to 'covered' probability mass
        bottom_model, bottom_w, top_models, filtered_out = o

        if filtered_out:
            # This outcome branch is already filtered out, can add as-is right
            # away
            filtered_outcomes.append(o)
            continue

        # Set of atoms actually covered by the outcome
        top_atoms = top_models.atoms() if top_models is not None else set()
        o_atoms = bottom_model | top_atoms

        # Disjunction to satisfy; making a copy that we can manipulate within this
        # for loop
        disjunction = {l for l in literals}

        add_as_is = False
        for l in literals:
            # literal shouldn't be covered by both bottom_model and top_models
            assert not ((l.as_atom() in bottom_model) and (l.as_atom() in top_atoms))

            if l.naf:
                ## l.naf==True; l is negative literal
                # If not covered by either bottom_model or top_models, the disjunction
                # is trivially satisfiable and this outcome can be included as-is without
                # filtering
                if l.as_atom() not in o_atoms:
                    add_as_is = True

                # If covered by bottom_models, this literal is never satisfiable and can
                # be removed from disjunction
                if l.as_atom() in bottom_model:
                    disjunction.remove(l)

            else:
                ## l.naf==False; l is positive literal
                # If covered by bottom_models, the disjunction is trivially satisfiable
                # and this outcome can be included as-is without filtering
                if l.as_atom() in bottom_model:
                    add_as_is = True

                # If not covered by either bottom_model or top_models, this literal is
                # never satisfiable and can be removed from disjunction
                if l.as_atom() not in o_atoms:
                    disjunction.remove(l)

        if add_as_is:
            # No filtering needed, add the outcome as-is
            filtered_outcomes.append(o)
            continue
        if len(disjunction) == 0:
            # Empty disjunction cannot be satisfied; flag as filtered and add to
            # outcome list
            filtered_outcomes.append((bottom_model, bottom_w, top_models, True))
            continue

        # Outcomes that reached here represent mixture of models satisfying the
        # disjunction and those that do not; need top filtering
        filtered_outcomes.append(
            (bottom_model, bottom_w, filter(top_models, literals), filtered_out)
        )

    return Models(outcomes=filtered_outcomes)
