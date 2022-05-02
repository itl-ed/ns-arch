"""
Implements LP^MLN model set class
"""
from collections import defaultdict

import numpy as np

from .literal import Literal
from .rule import Rule


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
                # Aggregate literals; probabilities are summed in logit-space and then
                # sigmoid-ed back to probability space
                lit_factors = [
                    f for f in self.factors
                    if type(f) == tuple and isinstance(f[0], Literal)
                ]
                non_lit_factors = [
                    f for f in self.factors
                    if not (type(f) == tuple and isinstance(f[0], Literal))
                ]

                lits_agg = defaultdict(lambda: (0.5, None))
                for lit, w_pr, coll in lit_factors:
                    w_pr_agg = _sigmoid(_logit(lits_agg[lit][0]) + _logit(w_pr))
                    coll_agg = lits_agg[lit][1] if coll is None else coll
                    lits_agg[lit] = (w_pr_agg, coll_agg)

                self.factors = [
                    (lit, w_pr, coll) for lit, (w_pr, coll) in lits_agg.items()
                ] + non_lit_factors

    def __repr__(self):
        descr = ""
        if self.factors is not None:
            descr = f"factors(len={len(self.factors)})"
        if self.outcomes is not None:
            descr = f"outcomes(len={len(self.outcomes)})"
        return f"Models({descr})"
    
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
                    assert len(f) == 3 and isinstance(f[0], Literal)
                    if f[2] is None or f[2] == True:
                        atoms.add(f[0])
            
            return atoms

        if self.outcomes is not None:
            atoms = set()
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program, and a Models instance for some
                # program top
                _, bottom_model, top_models = o
                atoms |= set(bottom_model)
                atoms |= top_models.atoms() if top_models is not None else set()
            
            return atoms

        raise ValueError("Invalid Models instance")

    def marginals(self, norm=True):
        """
        Compute and return marginals by aggregating total probability mass of models
        (covered by the Models instance) containing each grounded literal. If norm=True,
        normalize the marginal values by the total probability mass covered by this
        Models instance; in effect, this leads to ignoring possible models not covered
        by this instance and conditioning on the covered models.
        """
        if self.factors is not None:
            marginals = {}
            pmass_covered = 1.0

            for f in self.factors:
                if isinstance(f, Models):
                    # Recurse; another Models instance
                    f_marginals, f_pmass = f.marginals()
                    marginals = {
                        **marginals,
                        **f_marginals
                    }
                    pmass_covered *= f_pmass
                else:
                    # Base case; independent atoms with probability
                    assert len(f) == 3 and isinstance(f[0], Literal)
                    if f[2] is None:
                        # Not collapsed into either true or false
                        marginals[f[0]] = f[1]
                    else:
                        # Collapsed into either true or false
                        if f[2] is True:
                            marginals[f[0]] = 1.0
                            pmass_covered *= f[1]
                        else:
                            marginals[f[0]] = 0.0
                            pmass_covered *= (1 - f[1])
                    
            marginals = { k: v * pmass_covered for k, v in marginals.items() }

            if norm:
                marginals = { k: v / pmass_covered for k, v in marginals.items() }

            return marginals, pmass_covered
                    
        if self.outcomes is not None:
            marginals = defaultdict(float)
            pmass_covered = 0
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program (tree actually), and a Models
                # instance for some program top
                bottom_pr, bottom_model, top_models = o

                if top_models is None:
                    top_marginals = {}
                else:
                    assert isinstance(top_models, Models)
                    top_marginals, top_pmass = top_models.marginals(norm=False)

                if bottom_model is not None:
                    pmass_covered += bottom_pr * top_pmass

                    # Increment marginals of bottom model literals
                    for lit in bottom_model:
                        marginals[lit] += bottom_pr * top_pmass

                    # Increment conditionals of top model literals multiplied by marginals
                    # of bottom model literals (hence amounting to joint probabilities)
                    for lit, pr in top_marginals.items():
                        marginals[lit] += bottom_pr * pr

            if norm:
                marginals = { k: v / pmass_covered for k, v in marginals.items() }

            return dict(marginals), pmass_covered

        raise ValueError("Invalid Models instance")
    
    def filter(self, literal):
        """
        Filter the Models instance to return a new Models instance representing the
        subset of models satisfying the condition specified by the literal; positive
        literal must be contained, and negative literal must not.
        """
        assert type(literal) == Literal
        
        if self.factors is not None:
            if all([f is None for f in self.factors]):
                # Empty Models instance
                return self

            if literal.as_atom() not in self.atoms():
                # If literal atom is not covered by any factor
                if literal.naf == False:
                    # Postive literal; literal unsatisfiable, filter all
                    return Models(factors=[None for _ in self.factors])
                else:
                    # Negative literal; no models filtered out
                    return self

            filtered_factors = []
            for f in self.factors:
                if isinstance(f, Models):
                    # Recurse; another Models instance
                    if literal.as_atom() in f.atoms():
                        # Filter the factor Models instance
                        filtered_factors.append(f.filter(literal))
                    else:
                        # Irrelevant factor
                        filtered_factors.append(f)
                else:
                    # Base case; independent atoms with probability
                    assert len(f) == 3 and isinstance(f[0], Literal)

                    if f[0] == literal.as_atom():
                        # Collapse into matching truth value
                        # (Recall literal.naf==False: positive, literal.naf==True: negative)
                        filtered_factors.append((f[0], f[1], not literal.naf))
                    else:
                        # Irrelevant factor
                        filtered_factors.append(f)
            
            return Models(factors=filtered_factors)

        if self.outcomes is not None:
            if literal.as_atom() not in self.atoms():
                # If literal atom is not covered by any factor
                if literal.naf == False:
                    # Postive literal; literal unsatisfiable, filter all
                    return Models(outcomes=[
                        (o, None, None) for o in self.outcomes
                    ])
                else:
                    # Negative literal; no models filtered out
                    return self

            filtered_outcomes = []
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program (tree actually), and a Models
                # instance for some program top
                bottom_pr, bottom_model, top_models = o

                top_atoms = top_models.atoms()

                # literal shouldn't be covered by both bottom_model and top_models
                assert not ((literal.as_atom() in bottom_model) and (literal.as_atom() in top_atoms))

                if literal.naf == False:
                    # Postive literal
                    if literal.as_atom() in bottom_model:
                        # Covered by bottom model, add to filtered_outcomes intact
                        filtered_outcomes.append(o)

                    elif literal.as_atom() in top_atoms:
                        # Covered by top_models, add to filtered_outcomes with top filtered
                        filtered_outcomes.append(
                            (bottom_pr, bottom_model, top_models.filter(literal))
                        )

                    else:
                        # literal not covered by this outcome, filter out this outcome while
                        # leaving trace of it; (bottom_model is None) marks this filtering
                        filtered_outcomes.append((bottom_pr, None, None))
                else:
                    # Negative literal
                    if literal.as_atom() in bottom_model:
                        # literal covered by this outcome, filter out this outcome while
                        # leaving trace of it; (bottom_model is None) marks this filtering
                        filtered_outcomes.append((bottom_pr, None, None))

                    elif literal.as_atom() in top_atoms:
                        # Covered by top_models, add to filtered_outcomes with top filtered
                        filtered_outcomes.append(
                            (bottom_pr, bottom_model, top_models.filter(literal))
                        )

                    else:
                        # Not covered by either, add to filtered_outcomes intact
                        filtered_outcomes.append(o)
                
            return Models(outcomes=filtered_outcomes)

        raise ValueError("Invalid Models instance")
    
    def query_yn(self, event):
        """
        Query the tree structure to estimate the likelihood of specified event.
        Compute and return the aggregate probability mass of models which are
        also models satisfying the provided event, coded as a conjunction of
        literals.
        """
        if type(event) != set:
            try:
                # Treat as set
                event = set(event)
            except TypeError:
                # Accept single-rule event and wrap in a set
                assert isinstance(event, Literal)
                event = set([event])

        filtered = self
        for ev_lit in event:
            # Models satisfying event literal
            filtered = filtered.filter(ev_lit)

        # Ratio of total probability mass of models satisfying the event to that
        # of all models covered by this Models instance
        ev_prob = filtered.marginals()[1] / self.marginals()[1]

        return ev_prob


def _logit(p):
    """ Compute logit of the probability value p """
    if p == 1:
        return float("inf")
    elif p == 0:
        return float("-inf")
    else:
        return np.log(p/(1-p))

def _sigmoid(l):
    """ Compute probability of the logit value l """
    if l == float("inf"):
        return 1
    elif l == float("-inf"):
        return 0
    else:
        return 1 / (1 + np.exp(-l))
