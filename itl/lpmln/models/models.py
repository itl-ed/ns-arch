import re
from itertools import product
from collections import defaultdict
from functools import reduce

from .unnorm_mass import compute_Z
from .filter import filter_models
from .query import query
from ..literal import Literal
from ..polynomial import Polynomial
from ..utils import cacheable


class Models:
    """
    Representation of sets of models, either as:
        1) Complete factorization by joint distributions of independent atoms, each
            with its marginal probability, or
        2) Flattened list of individual model specifications (outcomes) by set of
            true atoms (i.e. Herbrand interpretation), each with its joint probability
    """
    def __init__(self, factors=None, outcomes=None):
        assert not ((factors is not None) and (outcomes is not None)), \
            "Do not provide both factors & outcomes as arg"
        if factors is not None:
            # Can flatten factors that are themselves single-factor Models instances
            while any(_is_single_factor_models(f) for f in factors):
                factors_flattened = []
                for f in factors:
                    if _is_single_factor_models(f):
                        factors_flattened.append(f.factors[0])
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
        if outcomes is not None:
            self.outcomes = outcomes

        # If single-factor instance and the single factor is itself a Models
        # instance, flatten the layer
        while _is_single_factor_models(self) and isinstance(self.factors[0], Models):
            if hasattr(self.factors[0], "factors"):
                self.factors = self.factors[0].factors
            elif hasattr(self.factors[0], "outcomes"):
                self.outcomes = self.factors[0].outcomes
                delattr(self, "factors")

        self.cache = {
            method: {} for method in dir(self)
            if callable(getattr(self, method)) and not method.startswith("__")
        }

    def __repr__(self):
        descr = ""
        if hasattr(self, "factors"):
            descr = f"factors(len={len(self.factors)})"
        if hasattr(self, "outcomes"):
            descr = f"outcomes(len={len(self.outcomes[1]) if self.outcomes[1] is not None else '*'})"
        return f"Models({descr})"

    @cacheable
    def is_empty(self):
        """
        Models instance is empty if it doesn't cover any models
        (Caution: instance with empty factor list is not empty, but indeed
        covering an empty model, thus not empty as Models instance)
        """
        if not (hasattr(self, "factors") or hasattr(self, "outcomes")):
            # Not having either signifies empty instance
            return True
        
        if hasattr(self, "outcomes") and len(self.outcomes)==0:
            # Instance with empty outcomes list is essentially an empty one
            return True

        return False

    def enumerate(self):
        """
        Unroll the Models instance to generate every single model covered, along with
        its joint probability.

        (TODO?: Let's see later if we can come up with a more 'reactive' algorithm that
        doesn't have to wait results from inner recursive calls)
        """
        factors_enums = []
        if hasattr(self, "factors"):
            for f in self.factors:
                if isinstance(f, Models):
                    # Recurse; another Models instance
                    factors_enums.append(set(f.enumerate()))
                else:
                    # Base case; independent atoms with probability
                    assert len(f) == 3
                    if f[0] is None:
                        # Signals violation of integrity constraint; disregard
                        continue

                    assert isinstance(f[0], Literal)

                    lit_p = (frozenset([f[0]]), f[1])
                    lit_n = (frozenset(), 1-f[1])

                    if f[2] is None:
                        factors_enums.append({lit_p, lit_n})
                    else:
                        if f[2]:
                            factors_enums.append({lit_p})
                        else:
                            factors_enums.append({lit_n})

            if len(factors_enums) > 0:
                for model_choices in product(*factors_enums):
                    yield (
                        frozenset.union(*[mc[0] for mc in model_choices]),
                        reduce(lambda p1,p2: p1*p2, [mc[1] for mc in model_choices])
                    )

        if hasattr(self, "outcomes"):
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program (tree actually), a weight sum
                # applied to the bottom model, a Models instance for some program top,
                # and a boolean flag denoting whether this outcome branch is filtered
                # out and thus shouldn't contribute to 'covered' probability mass
                bottom_model, bottom_w, top_models, filtered_out = o

                # This part has not been updated to comply with recent changes, and we
                # don't know what would happen here...
                raise NotImplementedError

                if top_models is None:
                    yield (frozenset(bottom_model), bottom_pr)
                else:
                    for top_model, top_pr in top_models.enumerate():
                        yield (frozenset(bottom_model | top_model), bottom_pr*top_pr)

    @cacheable
    def atoms(self):
        """ Return all atoms occurring in models covered by instance """
        if self.is_empty():
            # Empty Models instance
            return set()

        if hasattr(self, "factors"):
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

        if hasattr(self, "outcomes"):
            bottom_models, branch_consequences = self.outcomes

            if branch_consequences is None:
                atoms = bottom_models.atoms()
            else:
                atoms = set()
                for bc in branch_consequences:
                    # Each outcome entry is a tuple of a model for some program bottom in
                    # the splitting sequence of the program (tree actually), a weight sum
                    # applied to the bottom model, a Models instance for some program top,
                    # and a boolean flag denoting whether this outcome branch is filtered
                    # out and thus shouldn't contribute to 'covered' probability mass
                    branch_event, bm_filtered, _, _, top_models, filtered_out = bc
                    if filtered_out:
                        # Disregard this outcome branch
                        continue

                    atoms |= bm_filtered.atoms()
                    atoms |= {l for l in branch_event if l.naf==False}
                    atoms |= top_models.atoms() if top_models is not None else set()

            return atoms

        raise ValueError("Invalid Models instance")

    @cacheable
    def compute_Z(self):
        """
        Compute and return total unnormalized probability mass covered by this instance
        """
        return compute_Z(self)[1]

    @cacheable
    def filter(self, literals):
        """
        Filter the Models instance to return a new Models instance representing the
        subset of models satisfying the condition specified by the *disjunction* of
        literals, so that formulas in conjunctive normal form can be processed with
        a chain of calls to filter.
        """
        return filter_models(self, literals)

    @cacheable
    def query(self, q_vars, event, per_assignment=True, per_partition=False):
        """
        Query the tree structure to estimate the likelihood of each possible answer
        to the provided question, represented as tuple of entities (empty tuple for
        y/n questions). For each entity tuple that have some possible models satisfying
        the provided event specification, compute and return the aggregate probability
        mass of such models. Return the Models instances obtained after appropriate
        filtering as well.

        If q_vars is None we have a yes/no (polar) question, where having a non-empty
        tuple as q_vars indicates we have a wh-question.
        """
        return query(self, q_vars, event, per_assignment, per_partition)


def _is_single_factor_models(f):
    """ Tests if f is a factors-Models instance with only a single factor """
    return isinstance(f, Models) and hasattr(f, "factors") and len(f.factors)==1
