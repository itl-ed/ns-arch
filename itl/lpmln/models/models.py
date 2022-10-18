import operator
from collections import defaultdict
from itertools import product
from functools import reduce

from .common_factors import ModelsCommonFactors
from .branch_outcomes import ModelsBranchOutcomes
from .query import query
from ..polynomial import Polynomial
from ..utils import cacheable


class Models:
    """
    Representation of sets of models, which are probabilistic answer sets to an LP^MLN
    program with one or more topmost nodes. Each complete Models instance consists of
    a list of (ModelsCommonFactors, ModelsBranchOutcomes) pairs, each corresponding to
    a collection of models for some LP^MLN program with a single topmost node. The list
    may have zero, one or many elements depending of the dependency structure of the
    original LP^MLN program for which this Models instance represents valid answer sets
    (along with their probabilistic weights).
    """
    def __init__(self, factors_outcomes_pairs=None):
        assert all(len(cfs_bos)==2 for cfs_bos in factors_outcomes_pairs)
        assert all(
            isinstance(cfs, ModelsCommonFactors) or cfs is None
            for cfs, _ in factors_outcomes_pairs
        )
        assert all(
            isinstance(bos, ModelsBranchOutcomes) or bos is None
            for _, bos in factors_outcomes_pairs
        )

        if factors_outcomes_pairs is None:
            factors_outcomes_pairs = []
        self.cfs_bos_pairs = factors_outcomes_pairs

        self.cache = {
            method: {} for method in dir(self)
            if callable(getattr(self, method)) and not method.startswith("__")
        }

    def __repr__(self):
        return f"Models(len={len(self.cfs_bos_pairs)})"

    def is_empty(self):
        """
        Instance is empty if its list of factors-outcomes pairs is empty
        """
        return len(self.cfs_bos_pairs) == 0

    @cacheable
    def atoms(self):
        """ Return all atoms occurring covered by instance """
        if self.is_empty():
            # Empty Models instance, return empty set
            return set()

        atoms = set()
        for cfs, bos in self.cfs_bos_pairs:
            if bos is None:
                # List of branch outcomes nonexistent; full consideration of all
                # possible events from the bottom factors
                atoms |= cfs.atoms()
            else:
                # Only consider cases actually covered by branches
                atoms |= bos.atoms()
            
        return atoms

    def enumerate(self):
        """
        Unroll the instance to generate every single model covered, along with its
        total weight
        """
        if self.is_empty():
            # Empty Models instance, yield nothing
            raise StopIteration

        for cfs, bos in self.cfs_bos_pairs:
            if bos is None:
                # List of branch outcomes nonexistent; full consideration of all
                # possible events from the bottom factors
                yield from cfs.enumerate()
            else:
                # Only consider cases actually covered by branches
                for bottom_model, top_model in bos.enumerate():
                    yield (bottom_model[0] | top_model[0], bottom_model[1] * top_model[1])

    @cacheable
    def compute_Z(self):
        """
        Compute and return total unnormalized probability mass covered by this instance
        """
        if self.is_empty():
            # Empty Models instance, return zero
            return Polynomial(float_val=0.0)

        Z = Polynomial(float_val=1.0)
        for cfs, bos in self.cfs_bos_pairs:
            if bos is None:
                # List of branch outcomes nonexistent; full consideration of all
                # possible events from the bottom factors
                Z = Z * cfs.compute_Z()
            else:
                # Only consider cases actually covered by branches
                Z = Z * bos.compute_Z()

        # if len(self.cfs_bos_pairs) == 1 and self.cfs_bos_pairs[0][1] is None:
        #     # Simpler case with single-head Models instance without branching outcomes
        #     return self.cfs_bos_pairs[0][0].compute_Z()

        # outcomes_per_pair = [[] for _ in range(len(self.cfs_bos_pairs))]

        # for i, (cfs, bos) in enumerate(self.cfs_bos_pairs):
        #     if bos is None:
        #         # List of branch outcomes nonexistent; full consideration of all
        #         # possible events from the bottom factors
        #         raise NotImplementedError
        #     else:
        #         # Only consider cases actually covered by branches
        #         for bottom_model, top_model in bos.enumerate():
        #             outcomes_per_pair[i].append((bottom_model, top_model))

        # # Set of all bottom_models occurred
        # all_bottom_models = set.union(*[
        #     set(b_atms for (b_atms, _), (_, _) in outcomes)
        #     for outcomes in outcomes_per_pair
        # ])

        # # Universe of all possible positive atoms in bottom models per pair
        # bottom_univs_per_pair = [
        #     frozenset.union(*[b_atms for (b_atms, _), (_, _) in outcomes])
        #     for outcomes in outcomes_per_pair
        # ]

        # # Universe of all possible positive atoms in across all pairs
        # bottom_univ = frozenset.union(*bottom_univs_per_pair)

        # # Expand the witnessed all_bottom_models to all possible full instantiations
        # all_bottom_models_full = set()
        # for b_atms in all_bottom_models:
        #     rest_possible_insts = product(*{
        #         (atm, atm.flip()) for atm in bottom_univ - b_atms
        #     })
        #     for r_inst in rest_possible_insts:
        #         r_inst = frozenset(r_inst)
        #         all_bottom_models_full.add(b_atms | r_inst)

        # # Finalize the set of full models
        # all_models = defaultdict(list)
        # for i, outcomes in enumerate(outcomes_per_pair):
        #     pair_bottom_univ = bottom_univs_per_pair[i]
        #     bottom_branches = [
        #         (b_ev, b_Z)
        #         for b_ev, _, b_Z, _, _, _ in self.cfs_bos_pairs[i][1].outcomes
        #     ]

        #     for (b_atms, b_Z), (t_atms, t_Z) in outcomes:
        #         # Full specification of b_atms with respect to pair_bottom_univ
        #         b_lits_pfull = b_atms | {atm.flip() for atm in pair_bottom_univ-b_atms}

        #         # Possible full instantiations of b_atms
        #         b_lits_full_insts = {
        #             bm for bm in all_bottom_models_full if b_lits_pfull <= bm
        #         }
        #         for b_lits_full in b_lits_full_insts:
        #             all_models[b_lits_full].append((
        #                 b_Z, t_atms, t_Z, pair_bottom_univ, bottom_branches
        #             ))

        # # Collect Z from the final set of full models
        # Z = Polynomial(float_val=0.0)
        # for b_lits_full, consequences in all_models.items():
        #     # How do we ensure we avoided duplicate counting of probability masses
        #     # for the bottom branch event...? Here's an attempt...

        #     # The goal is to incrementally build the bottom base Z until the
        #     # whole bottom universe is covered
        #     univ_covered = set()
        #     bottom_base_Z = Polynomial(float_val=1.0)

        #     while len(univ_covered) < len(bottom_univ):
        #         # Incrementally cover the whole universe, each time by looking
        #         # for a consequence item that would best achieve it
        #         remainder_to_cover = bottom_univ - univ_covered
        #         consq_best_overlap = max(
        #             consequences, key=lambda c: len(c[3] & remainder_to_cover)
        #         )
        #         actual_overlap = consq_best_overlap[3] & remainder_to_cover

        #         # Reference against which marginal contribution of covering the
        #         # universe will be obtained
        #         b_lits_full_compare = frozenset({
        #             lit.flip() if lit.as_atom() in actual_overlap else lit
        #             for lit in b_lits_full
        #         })

        #         event_Z = [
        #             b_Z for b_ev, b_Z in consq_best_overlap[4]
        #             if b_ev <= b_lits_full
        #         ][0]
        #         reference_Z = [
        #             b_Z for b_ev, b_Z in consq_best_overlap[4]
        #             if b_ev <= b_lits_full_compare
        #         ][0]
        #         marginal_Z = event_Z / reference_Z

        #         univ_covered |= actual_overlap
        #         bottom_base_Z = bottom_base_Z * marginal_Z

        #     # Handling top Z values is easy, just multiply them altogether
        #     total_top_Z = reduce(operator.mul, [c[2] for c in consequences])

        #     Z += bottom_base_Z * total_top_Z

        return Z

    @cacheable
    def filter(self, literals):
        """
        Filter the Models instance to return a new Models instance representing the
        subset of models satisfying the condition specified by the *disjunction* of
        literals, so that formulas in conjunctive normal form can be processed with
        a chain of calls to filter.
        """
        if self.is_empty():
            # Empty Models instance, simply return self
            return self
        
        filtered_pairs = []
        for cfs, bos in self.cfs_bos_pairs:
            if bos is None:
                # List of branch outcomes nonexistent; full consideration of all
                # possible events from the bottom factors
                relevant_literals = literals & cfs.atoms()
                if len(relevant_literals) > 0:
                    filtered_pairs.append((cfs.filter(relevant_literals), bos))
                else:
                    filtered_pairs.append((cfs, bos))
            else:
                # Only consider cases actually covered by branches
                relevant_literals = literals & bos.atoms()
                if len(relevant_literals) > 0:
                    filtered_pairs.append((cfs, bos.filter(relevant_literals)))
                else:
                    filtered_pairs.append((cfs, bos))

        return Models(filtered_pairs)

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
