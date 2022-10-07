""" Models().compute_marginals() factored out """
from collections import defaultdict

import numpy as np

from ..literal import Literal
from ..utils import logit, sigmoid
from ..utils.polynomial import *


def compute_marginals(models):
    """
    Compute and return marginals for atoms covered by this instance
    """
    if models.is_empty():
        # Empty model, return None and zero
        return None, 0.0

    data_agg = {}
    _collect_unnorms(models, data_agg)

    marginals = {
        lit: (poly_ratio_at_limit(unnorm, Z), unnorm, Z)
        for lit, (unnorm, Z) in data_agg.items()
    }
    
    return marginals

def _collect_unnorms(models, data_agg):
    """
    Collect data to update structures that contain data needed for computing marginal
    probabilities
    """
    from .models import Models
    # An unnormalized probability mass is expressed as a dict, which represents an
    # appropriate polynomial including terms with different degrees where exp("a")
    # is considered to be variable. I.e. an empty dict ({}) would correspond to
    # value of zero, and { 0: 0.0 } would correspond to value of one. 1+exp("a")
    # would be represented as { 0: 0.0, 1: 0.0 }.
    # Each factor contributes to the partition function in a multiplicative basis,
    # and thus coefficients in Z can get very large -- so keep coefficients in
    # log space.

    if models.is_empty():
        # Empty Models instance
        return

    if hasattr(models, "factors"):
        results_per_factor = []
        for f in models.factors:
            if isinstance(f, Models):
                # Recurse; another Models instance
                f_unnorm_marginals, f_Z, f_Z_covered = _collect_unnorms(f)

                if len(f_Z) == 0:
                    # The factor covers no models at all; cannot make any conclusion
                    # about potentially covered atoms, disregard this factor
                    continue

                results_per_factor.append((f_unnorm_marginals, f_Z, f_Z_covered))
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3 and isinstance(f[0], Literal)
                f_lit, f_w, f_coll = f

                if f_w is not None:
                    # Partition function Z is 1+exp(f_w)
                    if type(f_w) == float:
                        # Soft weight
                        if f_coll is None:
                            Z = { 0: np.logaddexp(0, f_w) }
                            f_lit_unnorm = { 0: f_w }
                        else:
                            Z = { 0: f_w } if f_coll else { 0: 0.0 }
                            f_lit_unnorm = Z if f_coll else {}
                    else:
                        # Hard weight
                        assert type(f_w) == str
                        if f_w == "a":
                            if f_coll is None:
                                Z = { 0: 0.0, 1: 0.0 }
                                f_lit_unnorm = { 1: 0.0 }
                            else:
                                Z = { 1: 0.0 } if f_coll else { 0: 0.0 }
                                f_lit_unnorm = Z if f_coll else {}
                        elif f_w == "-a":
                            if f_coll is None:
                                Z = { 0: 0.0, -1: 0.0 }
                                f_lit_unnorm = { -1: 0.0 }
                            else:
                                Z = { -1: 0.0 } if f_coll else { 0: 0.0 }
                                f_lit_unnorm = Z if f_coll else {}
                        else:
                            raise ValueError("Invalid weight value")
                else:
                    # None-weight signifies 'absolute' (typically definitionally
                    # derived) facts, which do not concern model weights at all
                    print(0)
                
                data_agg[f_lit] = (f_lit_unnorm, Z)

        return

    if hasattr(models, "outcomes"):
        results_per_outcome = []
        for o in models.outcomes:
            # Each outcome entry is a tuple of a model for some program bottom in
            # the splitting sequence of the program (tree actually), a weight sum
            # applied to the bottom model, a Models instance for some program top,
            # and a boolean flag denoting whether this outcome branch is filtered
            # out and thus shouldn't contribute to 'covered' probability mass
            bottom_model, bottom_w, top_models, filtered_out = o
            bottom_w = { bottom_w[1]: bottom_w[0] }

            assert isinstance(top_models, Models)
            top_out = _collect_unnorms(top_models)

            results_per_outcome.append((bottom_model, bottom_w, top_out, filtered_out))

        return "o", results_per_outcome

    raise ValueError("Invalid Models instance")
