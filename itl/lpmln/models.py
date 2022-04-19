"""
Implements LP^MLN model set class
"""
from collections import defaultdict

import numpy as np

from .literal import Literal


class Models:
    """
    Representation of sets of models, either as:
        1) Complete factorization by joint distributions of independent atoms, each
            with its marginal probability, or
        2) Flattened list of individual model specifications (outcomes) by set of
            true atoms (i.e. Herbrand interpretation), each with its joint probability
    """
    def __init__(self, factors=None, outcomes=None):
        assert (factors is None) ^ (outcomes is None), \
            "Provide only one of either: factors or outcomes"

        self.factors = factors
        self.outcomes = outcomes

        if self.factors is not None:
            # Can flatten factors that are factored Models themselves
            fm_factors = [
                f for f in self.factors
                if isinstance(f, Models) and f.factors is not None
            ]

            if len(fm_factors) > 1:
                flattened = sum([f.factors for f in fm_factors], [])
                non_fm_factors = [
                    f for f in self.factors
                    if not (isinstance(f, Models) and f.factors is not None)
                ]

                self.factors = flattened + non_fm_factors
            
            if len(self.factors) > 1:
                # Aggregate literals; probabilities are logsumexp-ed (then exp-ed back)
                lit_factors = [
                    f for f in self.factors
                    if type(f) == tuple and isinstance(f[0], Literal)
                ]
                non_lit_factors = [
                    f for f in self.factors
                    if not (type(f) == tuple and isinstance(f[0], Literal))
                ]

                lits_agg = defaultdict(lambda: float("-inf"))
                for lit, w_pr in lit_factors:
                    lits_agg[lit] = np.exp(np.logaddexp(lits_agg[lit], np.log(w_pr)))

                self.factors = [(lit, w_pr) for lit, w_pr in lits_agg.items()] + non_lit_factors

    def __repr__(self):
        descr = ""
        if self.factors is not None:
            descr = f"factors(len={len(self.factors)})"
        if self.outcomes is not None:
            descr = f"outcomes(len={len(self.outcomes)})"
        return f"Models({descr})"
    
    def marginals(self):
        """ Compute and return marginals for occurring atoms with recursion """
        if self.factors is not None:
            marginals = {}
            for f in self.factors:
                if isinstance(f, Models):
                    # Recurse; another Models instance
                    f_marginals = f.marginals()
                    marginals = {
                        **marginals,
                        **f_marginals
                    }
                else:
                    # Base case; independent atoms with probability
                    assert len(f) == 2 and isinstance(f[0], Literal)
                    marginals[f[0]] = f[1]

            return marginals
                    
        if self.outcomes is not None:
            marginals = defaultdict(float)
            bottom_pr_sum = 0
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program (tree actually), and a Models
                # instance for some program top
                (bottom_model, bottom_pr), top_models = o
                bottom_pr_sum += bottom_pr

                if top_models is None:
                    top_marginals = {}
                else:
                    assert isinstance(top_models, Models)
                    top_marginals = top_models.marginals()

                # Increment marginals of bottom model literals
                for lit in bottom_model:
                    marginals[lit] += bottom_pr

                # Increment conditionals of top model literals multiplied by marginals
                # of bottom model literals (hence amounting to joint probabilities)
                for lit, pr in top_marginals.items():
                    marginals[lit] += pr * bottom_pr

            # Need to normalize values with sum of bottom model probabilities; in effect,
            # this leads to ignoring the pruned low-probability models and conditioning
            # top model events on the considered bottom models only
            marginals = {
                lit: pr / bottom_pr_sum for lit, pr in marginals.items()
            }

            return dict(marginals)

        raise ValueError("Invalid Models instance")
    
    def query_yn(self, event, neg=False):
        """
        Query the tree structure to estimate the likelihood of specified event
        (represented as a conjunction of literals, or its negation)
        """
        if type(event) != set:
            try:
                # Treat as set
                event = set(event)
            except TypeError:
                # Accept single-literal event and wrap in a set
                event = set([event])

        if self.factors is not None:
            # If event cannot be fully covered by self.atoms(), no possibility of
            # finding models satisfying event; terminate early
            if event & self.atoms() != event:
                if neg:
                    return 1.0
                else:
                    return 0.0

            event_pr = 1.0             # Multiply for each match from 1.0
            for f in self.factors:
                if isinstance(f, Models):
                    # Recurse; another Models instance
                    # First inspect set of atoms covered by each factor
                    f_atoms = f.atoms()

                    # Query the factor for the maximal subset of event covered
                    subevent = event & f_atoms
                    event_pr *= f.query_yn(subevent)
                else:
                    # Base case; independent atoms with probability
                    assert len(f) == 2 and isinstance(f[0], Literal)
                    if f[0] in event:
                        event_pr *= f[1]
            
            if neg:
                return 1 - event_pr
            else:
                return event_pr
        
        if self.outcomes is not None:
            event_pr = 0.0             # Add for each match from 0.0
            bottom_pr_sum = 0
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program, and a Models instance for some
                # program top
                (bottom_model, bottom_pr), top_models = o
                bottom_pr_sum += bottom_pr

                if top_models is None:
                    top_atoms = set()
                else:
                    top_atoms = top_models.atoms()
                o_atoms = set(bottom_model) | top_atoms

                if event <= o_atoms:
                    top_subevent = event & top_atoms

                    if len(top_subevent) == 0:
                        # Do not need to dig further if top_subevent is empty
                        event_pr += bottom_pr
                    else:
                        # Multiply probabilities
                        top_pr = top_models.query_yn(top_subevent)
                        event_pr += bottom_pr * top_pr
            
            # Need to normalize values with sum of bottom model probabilities
            event_pr /= bottom_pr_sum
            
            if neg:
                return 1 - event_pr
            else:
                return event_pr

        raise ValueError("Invalid Models instance")

    def atoms(self):
        """ Return all atoms occurring in models covered by instance """
        if self.factors is not None:
            atoms = set()
            for f in self.factors:
                if isinstance(f, Models):
                    # Recurse; another Models instance
                    atoms |= f.atoms()
                else:
                    # Base case; independent atoms with probability
                    assert len(f) == 2 and isinstance(f[0], Literal)
                    atoms.add(f[0])
            
            return atoms

        if self.outcomes is not None:
            atoms = set()
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program, and a Models instance for some
                # program top
                (bottom_model, _), top_models = o
                atoms |= set(bottom_model)
                atoms |= top_models.atoms()
            
            return atoms

        raise ValueError("Invalid Models instance")
