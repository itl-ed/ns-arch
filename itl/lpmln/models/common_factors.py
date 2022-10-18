import operator
from itertools import product
from functools import reduce
from collections import defaultdict

from ..literal import Literal
from ..polynomial import Polynomial
from ..utils import cacheable


class ModelsCommonFactors:
    """
    Representation of a set of common factors which serve as the 'basis' of some
    collection of models (which are probabilistic answer sets to a LP^MLN program
    with a single topmost node). Pair with a parter ModelsBranchOutcomes instance
    to form a complete model collection for a single-top-node program.
    """
    def __init__(self, factors=None):
        if factors is None:
            # Empty factors would represent absence of basis from which branch
            # events would be generated
            factors = []
        else:
            # Can flatten factors that are themselves Models instances with a single
            # factor and None-BranchOutcomes
            while any(_is_single_factor_models(f) for f in factors):
                factors_flattened = []
                for f in factors:
                    if _is_single_factor_models(f):
                        factors_flattened.append(f.cfs_bos_pairs[0][0].factors[0])
                    else:
                        factors_flattened.append(f)
                factors = factors_flattened

            if len(factors) > 1:
                # Aggregate literal factors with same head; weights are summed in
                # log-space
                lit_factors = [
                    f for f in factors
                    if type(f) == tuple and isinstance(f[0], Literal)
                ]
                non_lit_factors = [
                    f for f in factors
                    if not (type(f) == tuple and isinstance(f[0], Literal))
                ]

                lits_agg = defaultdict(lambda: (Polynomial(float_val=1.0), None))
                for lit, w, coll in lit_factors:
                    if w is not None:
                        # Non-absolute (incidental) rule weight
                        w_agg = lits_agg[lit][0] * Polynomial.from_primitive(w)
                    else:
                        # Absolute rule weight
                        w_agg = None
                    coll_agg = lits_agg[lit][1] if coll is None else coll
                    lits_agg[lit] = (w_agg, coll_agg)

                factors = [
                    (lit, w.primitivize() if w is not None else None, coll)
                    for lit, (w, coll) in lits_agg.items()
                ] + non_lit_factors

            self.factors = factors
        
        self.cache = {
            method: {} for method in dir(self)
            if callable(getattr(self, method)) and not method.startswith("__")
        }

    def __repr__(self):
        return f"ModelsCommonFactors(len={len(self.factors)})"
    
    def is_empty(self):
        """
        Instance is empty if its list of factors is empty
        """
        return len(self.factors) == 0

    @cacheable
    def atoms(self):
        """ Return all atoms due to factors covered by instance """
        from .models import Models

        atoms = set()
        for f in self.factors:
            if isinstance(f, Models):
                # Recurse; another Models instance
                atoms |= f.atoms()
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3
                if f[0] is None:
                    # Signals violation of integrity constraint; disregard
                    continue

                assert isinstance(f[0], Literal)
                if f[2] is None or f[2] == True:
                    atoms.add(f[0])
        
        return atoms

    def enumerate(self):
        """
        Unroll the instance to generate every single model covered, along with its
        total weight
        """
        from .models import Models

        factors_enums = []
        for f in self.factors:
            if isinstance(f, Models):
                # Recurse; another Models instance
                factors_enums.append(f.enumerate())
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3
                f_lit, f_w, f_coll = f

                if f_lit is None:
                    # Signals violation of integrity constraint; disregard
                    continue
                assert isinstance(f[0], Literal)

                if f_w is not None:
                    # Non-absolute incidental rule, hard or soft
                    f_w_poly = Polynomial.from_primitive(f_w)
                    lit_p = (frozenset([f_lit]), f_w_poly)
                    lit_n = (frozenset(), Polynomial(float_val=1.0))

                    if f_coll is None:
                        factors_enums.append({lit_p, lit_n})
                    else:
                        if f_coll:
                            factors_enums.append({lit_p})
                        else:
                            factors_enums.append({lit_n})
                else:
                    # Absolute rule that doesn't allow failure to derive f_lit
                    lit_p = (frozenset([f_lit]), Polynomial(float_val=1.0))

                    if f_coll is None:
                        factors_enums.append({lit_p})
                    else:
                        if f_coll:
                            factors_enums.append({lit_p})
                        else:
                            factors_enums.append(set())

        if len(factors_enums) > 0:
            for model_choices in product(*factors_enums):
                yield (
                    frozenset.union(*[mc[0] for mc in model_choices]),
                    reduce(lambda p1,p2: p1*p2, [mc[1] for mc in model_choices])
                )
        else:
            # Empty factor list; single model (empty) and baseline weight
            yield (frozenset(), Polynomial(float_val=1.0))

    @cacheable
    def compute_Z(self):
        """
        Compute and return total unnormalized probability mass covered by this instance
        """
        from .models import Models

        Z = Polynomial(float_val=1.0)
        for f in self.factors:
            if isinstance(f, Models):
                # Recurse; another Models instance
                f_Z = f.compute_Z()
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3
                if f[0] is None:
                    # Signals violation of integrity constraint; disregard
                    continue

                assert isinstance(f[0], Literal)
                _, f_w, f_coll = f

                if f_w is not None:
                    # Partition function Z is 1+exp(f_w)
                    f_w_poly = Polynomial.from_primitive(f_w)

                    # Actually covered pmass
                    if f_coll is None:
                        f_Z = f_w_poly + Polynomial(float_val=1.0)
                    else:
                        if f_coll:
                            f_Z = f_w_poly
                        else:
                            f_Z = Polynomial(float_val=1.0)
                else:
                    # None-weight signifies 'absolute' (typically definitionally
                    # derived) facts, which do not even allow possibility of the
                    # literal not holding
                    if f_coll is None:
                        f_Z = Polynomial(float_val=1.0)
                    else:
                        if f_coll:
                            f_Z = Polynomial(float_val=1.0)
                        else:
                            f_Z = Polynomial(float_val=0.0)

            if f_Z is not None:
                Z = Z * f_Z

        return Z

    @cacheable
    def filter(self, literals):
        """
        Filter the instance to return a new instance representing the subset of
        models satisfying the condition specified by the *disjunction* of literals,
        so that formulas in conjunctive normal form can be processed with a chain of
        calls to filter.
        """
        from .models import Models

        # Sets of atoms actually covered by the factors
        f_atoms = [
            f.atoms() if isinstance(f, Models) else
                ({f[0]} if f[2] is None or f[2]==True else set())
            for f in self.factors
        ]

        # Relevant factor for each literal in disjunction
        lit_relevant_factors = {
            l: {i for i, atms in enumerate(f_atoms) if l.as_atom() in atms}
            for l in literals
        }

        for l in literals:
            if len(lit_relevant_factors[l])==0:
                if l.naf:
                    ## l.naf==True; l is negative literal
                    # If a negative literal in disjunction is not covered by any
                    # factor, the disjunction is trivially satisfiable and models
                    # can be returned right away
                    return self
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
        for i, f in enumerate(self.factors):
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
                    filtered_factors.append(
                        (f.filter(frozenset(covered_lits)), i)
                    )
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
                (ms, enum_f(self.factors[i])-ms) for ms, i in f_enum_models
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
                self.factors[i] for _, i in filtered_factors
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

        return ModelsCommonFactors(factors=filtered_factors+intact_factors)


def _is_single_factor_models(f):
    """ Tests if f is a factors-Models instance with only a single factor """
    from .models import Models
    return isinstance(f, Models) and len(f.cfs_bos_pairs)==1 and \
        f.cfs_bos_pairs[0][1] is None and len(f.cfs_bos_pairs[0][0].factors)==1
