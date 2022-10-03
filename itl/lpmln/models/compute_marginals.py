""" Models().compute_marginals() factored out """
from itertools import product
from collections import defaultdict

import numpy as np

from ..literal import Literal
from ..utils import logit, sigmoid
from ..utils.polynomial import *


def compute_marginals(models, norm=True):
    """
    Compute and return normalized marginals and total probability mass covered by
    this instance. If norm=True, normalize the marginal values by this total pmass;
    in effect, this amounts to ignoring possible models not covered by this instance
    and conditioning on the covered models.
    """
    if models.is_empty():
        # Empty model, return None and zero
        return None, 0.0

    unnorm_marginals, Z_covered, Z = _compute_unnorm_marginals(models)

    pmass_covered = poly_ratio_at_limit(Z_covered, Z)
    if pmass_covered == 0:
        # No probability mass covered at all, return empty marginals and zero
        return {}, 0.0
    else:
        # Some probability mass covered, compute normalized marginals
        marginals = {
            k: poly_ratio_at_limit(v, Z) for k, v in unnorm_marginals.items()
        }

        if norm:
            # Further normalization by pmass_covered
            marginals = {
                k: v / pmass_covered for k, v in marginals.items()
            }

        return marginals, pmass_covered

def _compute_unnorm_marginals(models):
    """
    Compute and return unnormalized marginals for atoms covered by this instance,
    total probability mass covered, and the partition function Z.
    """
    from .models import Models

    if models.is_empty():
        # Empty Models instance
        return None, {}, {}

    if hasattr(models, "factors"):
        # Marginal probability values by atom
        unnorm_marginals = {}

        # Partition function (denominator) expressed as a dict, which represents
        # appropriate polynomial including terms with different degrees where exp("a")
        # is considered to be variable. I.e. an empty dict ({}) would correspond to
        # value of zero, and { 0: 0.0 } would correspond to value of one. 1+exp("a")
        # would be represented as { 0: 0.0, 1: 0.0 }.
        # Each factor contributes to the partition function in a multiplicative basis,
        # and thus coefficients in Z can get very large -- so keep coefficients in
        # log space.
        Z = { 0: 0.0 }

        # Total unnormalized probability mass covered by this Models instance
        Z_covered = { 0: 0.0 }

        for f in models.factors:
            if isinstance(f, Models):
                # Recurse; another Models instance
                f_unnorm_marginals, f_Z_covered, f_Z = _compute_unnorm_marginals(f)

                if len(f_Z) == 0:
                    # The factor covers no models at all; cannot make any conclusion
                    # about potentially covered atoms, disregard this factor
                    continue

                unnorm_marginals = {
                    **{k: poly_prod(v, f_Z) for k, v in unnorm_marginals.items()},
                    **{k: poly_prod(v, Z) for k, v in f_unnorm_marginals.items()}
                }

                Z = poly_prod(Z, f_Z)
                Z_covered = poly_prod(Z_covered, f_Z_covered)
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3 and isinstance(f[0], Literal)
                f_lit, f_w, f_coll = f
                prev_Z = Z

                # Update partition function Z and adjust existing unnormalized marginals
                if f_w is not None:
                    if type(f_w) == float:
                        # Multiplying constant log(1+exp(w)) to Z
                        mult_poly = { 0: np.logaddexp(0, f_w) }
                    else:
                        # Multiplying polynomial log(1+exp(a))/log(1+exp(-a)) to Z
                        assert type(f_w) == str
                        if f_w == "a":
                            mult_poly = { 0: 0.0, 1: 0.0 }
                        elif f_w == "-a":
                            mult_poly = { 0: 0.0, -1: 0.0 }
                        else:
                            raise ValueError("Invalid weight value")
                    Z = poly_prod(Z, mult_poly)
                    unnorm_marginals = {
                        k: poly_prod(v, mult_poly) for k, v in unnorm_marginals.items()
                    }

                # Add new unnormalized marginals for the new atom, and update Z_covered
                # accordingly
                if f_w is not None:
                    # Not collapsed into either true or false
                    if type(f_w) == float:
                        mult_poly = { 0: f_w }
                    else:
                        assert type(f_w) == str
                        if f_w == "a":
                            mult_poly = { 1: 0.0 }
                        elif f_w == "-a":
                            mult_poly = { -1: 0.0 }
                        else:
                            raise ValueError("Invalid weight value")

                    if f_coll is None:
                        # Not collapsed into either true or false; Z_covered is updated
                        # just as much as Z is
                        unnorm_marginals[f_lit] = poly_prod(prev_Z, mult_poly)
                        Z_covered = poly_prod(Z_covered, poly_sum(mult_poly, { 0: 0.0 }))
                    else:
                        # Collapsed into either true or false
                        if f_coll is True:
                            unnorm_marginals[f_lit] = prev_Z
                        else:
                            unnorm_marginals[f_lit] = {}
                            # Flip multiplier polynomial appropriately
                            if 0 in mult_poly:
                                mult_poly = { 0: logit(1-sigmoid(mult_poly[0])) }
                            else:
                                mult_poly = { -k: v for k, v in mult_poly.items() }

                        Z_covered = poly_prod(Z_covered, mult_poly)
                else:
                    # None-weight signifies 'absolute' (typically definitionally
                    # derived) facts, which do not concern model weights at all
                    unnorm_marginals[f_lit] = prev_Z

        return unnorm_marginals, Z_covered, Z

    if hasattr(models, "outcomes"):
        unnorm_marginals = defaultdict(dict)
        Z_covered = {}
        Z = {}

        for o in models.outcomes:
            # Each outcome entry is a tuple of a model for some program bottom in
            # the splitting sequence of the program (tree actually), a weight sum
            # applied to the bottom model, a Models instance for some program top,
            # and a boolean flag denoting whether this outcome branch is filtered
            # out and thus shouldn't contribute to 'covered' probability mass
            bottom_model, bottom_w, top_models, filtered_out = o
            bottom_w = { bottom_w[1]: bottom_w[0] }

            assert isinstance(top_models, Models)
            top_unnorm_marginals, top_Z_covered, top_Z = _compute_unnorm_marginals(
                top_models
            )

            if top_unnorm_marginals is None: continue      # No models, nothing to add

            if bottom_model is None:
                raise ValueError("Would it ever get here?")

            # Adjusted Z for this outcome branch to add to aggregate Z
            outcome_Z = poly_prod(bottom_w, top_Z)
            Z = poly_sum(Z, outcome_Z)

            if not filtered_out:
                # Adjusted Z_covered for this outcome branch to add to aggregate
                # Z_covered, only if this branch is not filtered out
                outcome_Z_covered = poly_prod(bottom_w, top_Z_covered)
                Z_covered = poly_sum(Z_covered, outcome_Z_covered)

            # Increment marginals of bottom model literals
            for lit in bottom_model:
                for_lit = poly_prod(bottom_w, top_Z_covered)
                unnorm_marginals[lit] = poly_sum(unnorm_marginals[lit], for_lit)

            # Increment conditionals of top model literals multiplied by marginals
            # of bottom model literals (hence amounting to joint probabilities)
            for lit, unnorm_pr in top_unnorm_marginals.items():
                for_lit = poly_prod(bottom_w, unnorm_pr)
                unnorm_marginals[lit] = poly_sum(unnorm_marginals[lit], for_lit)

        if len(Z) == 0:
            # No models whatsoever covered by this instance
            return None, {}, {}

        return dict(unnorm_marginals), Z_covered, Z

    raise ValueError("Invalid Models instance")
