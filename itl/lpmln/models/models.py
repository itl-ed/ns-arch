import re
from itertools import product
from collections import defaultdict
from functools import reduce

from .unnorm_mass import compute_Z
from .compute_marginals import compute_marginals
from .filter import filter
from .query import query
from ..literal import Literal


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
            # Can flatten factors that are factored Models themselves
            fm_factors = [
                f for f in factors
                if isinstance(f, Models) and hasattr(f, "factors")
            ]

            if len(fm_factors) > 1:
                flattened = sum([f.factors for f in fm_factors], [])
                non_fm_factors = [
                    f for f in factors
                    if not (isinstance(f, Models) and hasattr(f, "factors"))
                ]

                factors = flattened + non_fm_factors
            
            if len(factors) > 1:
                # Aggregate literals; probabilities are summed in logit-space and then
                # sigmoid-ed back to probability space
                lit_factors = [
                    f for f in factors
                    if type(f) == tuple and isinstance(f[0], Literal)
                ]
                non_lit_factors = [
                    f for f in factors
                    if not (type(f) == tuple and isinstance(f[0], Literal))
                ]

                lits_agg = defaultdict(lambda: (0, None))
                for lit, w, coll in lit_factors:
                    w_agg = _sum_weights(lits_agg[lit][0], w)
                    coll_agg = lits_agg[lit][1] if coll is None else coll
                    lits_agg[lit] = (w_agg, coll_agg)

                factors = [
                    (lit, w, coll) for lit, (w, coll) in lits_agg.items()
                ] + non_lit_factors

        if factors is not None:
            # Can remove empty Models-factor without any effect
            factors = [
                f for f in factors
                if not (isinstance(f, Models) and f.is_empty())
            ]
            self.factors = factors
        if outcomes is not None:
            self.outcomes = outcomes

        # If single-factor instance and the single factor is itself a Models
        # instance, flatten the layer
        while hasattr(self, "factors") and \
            len(self.factors) == 1 and \
            isinstance(self.factors[0], Models):

            if hasattr(self.factors[0], "factors"):
                self.factors = self.factors[0].factors
            elif hasattr(self.factors[0], "outcomes"):
                self.outcomes = self.factors[0].outcomes
                delattr(self, "factors")

    def __repr__(self):
        descr = ""
        if hasattr(self, "factors"):
            descr = f"factors(len={len(self.factors)})"
        if hasattr(self, "outcomes"):
            descr = f"outcomes(len={len(self.outcomes)})"
        return f"Models({descr})"
    
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
                    assert len(f) == 3 and isinstance(f[0], Literal)

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
                    assert len(f) == 3 and isinstance(f[0], Literal)
                    if f[2] is None or f[2] == True:
                        atoms.add(f[0])
            
            return atoms

        if hasattr(self, "outcomes"):
            atoms = set()
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program (tree actually), a weight sum
                # applied to the bottom model, a Models instance for some program top,
                # and a boolean flag denoting whether this outcome branch is filtered
                # out and thus shouldn't contribute to 'covered' probability mass
                bottom_model, _, top_models, filtered_out = o
                if filtered_out:
                    # Disregard this outcome branch
                    continue

                atoms |= set(bottom_model)
                atoms |= top_models.atoms() if top_models is not None else set()
            
            return atoms

        raise ValueError("Invalid Models instance")

    def compute_Z(self):
        """
        Compute and return total unnormalized probability mass covered by this instance
        """
        return compute_Z(self)[1]

    def compute_marginals(self):
        """
        Compute and return marginals for atoms covered by this instance
        """
        return compute_marginals(self)

    def filter(self, literals):
        """
        Filter the Models instance to return a new Models instance representing the
        subset of models satisfying the condition specified by the *disjunction* of
        literals, so that formulas in conjunctive normal form can be processed with
        a chain of calls to filter.
        """
        return filter(self, literals)

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


def _sum_weights(w1, w2):
    """ Combine two weights, appropriately handling soft/hard ones """
    if w1 is None or w2 is None:
        # Either is absolute rule; treat aggregate as absolute as well
        return None

    if (type(w1)==float or type(w1)==int) and (type(w2)==float or type(w2)==int):
        # Sum of pure number weights
        return w1+w2

    elif type(w1)==str and type(w2)==str:
        # Sum of pure (multiples of) hard weights
        w1_parse = re.match("(-?)(\d*)(.+)", w1)
        w2_parse = re.match("(-?)(\d*)(.+)", w2)
        w1_sign, w1_mult, w1_hardRep = w1_parse.groups()
        w2_sign, w2_mult, w2_hardRep = w2_parse.groups()

        assert w1_hardRep == w2_hardRep
        w1_mult = int(w1_mult) if len(w1_mult)>0 else 1
        w2_mult = int(w2_mult) if len(w2_mult)>0 else 1
        w1_coeff = w1_mult if len(w1_sign)==0 else -w1_mult
        w2_coeff = w2_mult if len(w2_sign)==0 else -w2_mult

        sum_coeff = w1_coeff + w2_coeff
        if sum_coeff == 0:
            return 0
        elif sum_coeff == 1:
            return w1_hardRep
        elif sum_coeff == -1:
            return f"-{w1_hardRep}"
        else:
            return f"{sum_coeff}{w1_hardRep}"

    else:
        # Otherwise, cast w1 and w2 into generic tuple forms and compute sum
        w1 = _generalize_weight(w1)
        w2 = _generalize_weight(w2)
        assert w1[2] is None or w2[2] is None or w1[2]==w2[2]

        sum = (w1[0]+w2[0], w1[1]+w2[1], w1[2] or w2[2])
        if sum[0] == 0:
            return sum[1]
        elif sum[1] == 0:
            if sum[0] == 1:
                return sum[2]
            elif sum[0] == -1:
                return f"-{sum[2]}"
            else:
                return f"{sum[0]}{sum[2]}"
        else:
            return sum

def _generalize_weight(w):
    """
    Cast weight value into generic tuple form: (Hard weight multiplier, soft weight,
    hard weight representation (e.g. "a" for alpha))
    """
    if type(w)==float or type(w)==int:
        return (0, w, None)
    elif type(w)==str:
        w_parse = re.match("(-?)(\d*)(.+)", w)
        w_sign, w_mult, w_hardRep = w_parse.groups()

        w_mult = int(w_mult) if len(w_mult)>0 else 1
        w_coeff = w_mult if len(w_sign)==0 else -w_mult

        return (w_coeff, 0, w_hardRep)
    else:
        assert type(w)==tuple       # Already in generic form
        return w
