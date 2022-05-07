"""
Implements LP^MLN model set class
"""
from itertools import product
from functools import reduce
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
            "Do not provide both factors & outcomes as arg"

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
    
    def is_empty(self):
        return (self.factors is None) and (self.outcomes is None)
    
    def atoms(self):
        """ Return all atoms occurring in models covered by instance """
        if self.is_empty():
            # Empty Models instance
            return set()

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
        (covered by the Models instance) containing each grounded literal. Also returns
        the total probability mass covered by this Models instance. If norm=True, normalize
        the marginal values by this total pmass; in effect, this amounts to ignoring
        possible models not covered by this instance and conditioning on the covered models.
        """
        if self.is_empty():
            # Empty Models instance
            return {}, 0.0

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

            if not norm:
                marginals = { k: v * pmass_covered for k, v in marginals.items() }

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
                    top_marginals, top_pmass = {}, 1.0
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
    
    def enumerate(self):
        """
        Unroll the Models instance to generate every single model covered, along with
        its joint probability.
        """
        factors_enums = []
        if self.factors is not None:
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
            
            for model_choices in product(*factors_enums):
                yield (
                    frozenset.union(*[mc[0] for mc in model_choices]),
                    reduce(lambda p1,p2: p1*p2, [mc[1] for mc in model_choices])
                )

        if self.outcomes is not None:
            for o in self.outcomes:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program (tree actually), and a Models
                # instance for some program top
                bottom_pr, bottom_model, top_models = o

                if top_models is None:
                    yield (frozenset(bottom_model), bottom_pr)
                else:
                    for top_model, top_pr in top_models.enumerate():
                        yield (frozenset(bottom_model | top_model), bottom_pr*top_pr)

    def filter(self, literals):
        """
        Filter the Models instance to return a new Models instance representing the
        subset of models satisfying the condition specified by the *disjunction* of
        literals, so that formulas in conjunctive normal form can be processed with
        a chain of calls to filter.
        """
        if self.is_empty():
            # Empty Models instance
            return self

        if type(literals) != set:
            try:
                # Treat as set
                literals = set(literals)
            except TypeError:
                # Accept single-literal disjunction and wrap in a set
                assert isinstance(literals, Literal)
                literals = {literals}

        if self.factors is not None:
            return self._filter_factors(literals)

        if self.outcomes is not None:
            return self._filter_outcomes(literals)

        raise ValueError("Invalid Models instance")

    def _filter_factors(self, literals):
        """
        Factorizing out submethod to be called by self.filter, for factors-Models
        instances
        """
        assert self.factors is not None

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
        # (Invariant: Each literal should have at most one relevant factor)
        assert all([len(s)<=1 for s in lit_relevant_factors.values()])

        for l in literals:
            if len(lit_relevant_factors[l])==0:
                if l.naf:
                    ## l.naf==True; l is negative literal
                    # If a negative literal in disjunction is not covered by any
                    # factor, the disjunction is trivially satisfiable and self
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
                    filtered_factors.append((f.filter(covered_lits), i))
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
                (ms, enum_f(self.factors[i])-ms) for ms, i in f_enum_models
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

    def _filter_outcomes(self, literals):
        """
        Factorizing out submethod to be called by self.filter, for outcomes-Models
        instances
        """
        assert self.outcomes is not None

        # Filter each outcome and add to new outcome list if result is not empty
        filtered_outcomes = []
        for o in self.outcomes:
            # Each outcome entry is a tuple of a model for some program bottom in
            # the splitting sequence of the program (tree actually), and a Models
            # instance for some program top
            bottom_pr, bottom_model, top_models = o

            # Set of atoms actually covered by the outcome
            top_atoms = top_models.atoms()
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
                # Empty disjunction cannot be satisfied; disregard outcome
                continue

            # Outcomes that reached here represent mixture of models satisfying the
            # disjunction and those that do not; need top filtering
            filtered_outcomes.append(
                (bottom_pr, bottom_model, top_models.filter(literals))
            )

        if len(filtered_outcomes) == 0:
            filtered_outcomes = None

        return Models(outcomes=filtered_outcomes)
    
    def query_yn(self, event):
        """
        Query the tree structure to estimate the likelihood of specified event.
        Compute and return the aggregate probability mass of models which are
        also models satisfying the provided event, coded as a set of rules.
        """
        if type(event) != set:
            try:
                # Treat as set
                event = set(event)
            except TypeError:
                # Accept single-rule event and wrap in a set
                assert isinstance(event, Rule)
                event = set([event])

        filtered = [self]
        for ev_rule in event:
            # Models satisfying rule head & body
            filtered_hb = filtered
            if len(ev_rule.head) > 0:
                for lits in [ev_rule.head] + ev_rule.body:
                    filtered_hb = [f.filter(lits) for f in filtered_hb]
            else:
                filtered_hb = []

            # Models not satisfying body
            filtered_nb = filtered
            if len(ev_rule.body) > 0:
                # Negation of conjunction of body literals == Disjunction of negated
                # body literals
                body_neg = [bl.flip() for bl in ev_rule.body]
                filtered_nb = [f.filter(body_neg) for f in filtered_nb]
            else:
                filtered_nb = []

            filtered = filtered_hb + filtered_nb

        # Ratio of total probability mass of models satisfying the event to that
        # of all models covered by this Models instance
        ev_prob = sum([fm.marginals()[1] for fm in filtered]) / self.marginals()[1]

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
