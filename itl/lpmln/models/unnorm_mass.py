""" Models().compute_marginals() factored out """
from ..literal import Literal
from ..utils.polynomial import *


def compute_Z(models):
    """
    Compute and return total unnormalized probability mass covered by this instance
    """
    from .models import Models

    if models.is_empty():
        # Empty model, return zero
        return {}, {}

    if hasattr(models, "factors"):
        Z = { 0: 0.0 }              # Represents 1.0
        Z_covered = { 0: 0.0 }      # Covered mass that are not filtered
        for f in models.factors:
            if isinstance(f, Models):
                # Recurse; another Models instance
                f_Z, f_Z_covered = compute_Z(f)
            else:
                # Base case; independent atoms with probability
                assert len(f) == 3 and isinstance(f[0], Literal)
                _, f_w, f_coll = f

                if f_w is not None:
                    # Partition function Z is 1+exp(f_w)
                    if type(f_w) == float:
                        # Soft weight
                        f_Z = { 0: np.logaddexp(0, f_w) }
                        if f_coll is None:
                            f_Z_covered = f_Z
                        else:
                            f_Z_covered = { 0: f_w } if f_coll else { 0: 0.0 }
                    else:
                        # Hard weight
                        assert type(f_w) == str
                        if f_w == "a":
                            f_Z = { 0: 0.0, 1: 0.0 }
                            if f_coll is None:
                                f_Z_covered = f_Z
                            else:
                                f_Z_covered = { 1: 0.0 } if f_coll else { 0: 0.0 }
                        elif f_w == "-a":
                            f_Z = { 0: 0.0, -1: 0.0 }
                            if f_coll is None:
                                f_Z_covered = f_Z
                            else:
                                f_Z_covered = { -1: 0.0 } if f_coll else { 0: 0.0 }
                        else:
                            raise ValueError("Invalid weight value")
                else:
                    # None-weight signifies 'absolute' (typically definitionally
                    # derived) facts, which do not concern model weights at all
                    f_Z = f_Z_covered = None

            if f_Z is not None:
                Z = poly_prod(Z, f_Z)
                Z_covered = poly_prod(Z_covered, f_Z_covered)
        
        return Z, Z_covered

    if hasattr(models, "outcomes"):
        Z = {}              # Represents 0.0
        Z_covered = {}      # Covered mass that are not filtered
        for o in models.outcomes:
            # Each outcome entry is a tuple of a model for some program bottom in
            # the splitting sequence of the program (tree actually), a weight sum
            # applied to the bottom model, a Models instance for some program top,
            # and a boolean flag denoting whether this outcome branch is filtered
            # out and thus shouldn't contribute to 'covered' probability mass
            _, bottom_w, top_models, filtered_out = o
            bottom_w = { bottom_w[1]: bottom_w[0] }

            assert isinstance(top_models, Models)
            top_Z, top_Z_covered = compute_Z(top_models)

            if top_Z is not None:
                o_Z = poly_prod(bottom_w, top_Z)
                o_Z_covered = poly_prod(bottom_w, top_Z_covered)
            else:
                o_Z = bottom_w

            Z = poly_sum(Z, o_Z)
            if not filtered_out:
                Z_covered = poly_sum(Z_covered, o_Z_covered)
        
        return Z, Z_covered

    raise ValueError("Invalid Models instance")
