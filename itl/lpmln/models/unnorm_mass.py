""" Models().compute_Z() factored out """
from ..literal import Literal
from ..polynomial import Polynomial


def compute_Z(models):
    """
    Compute and return total unnormalized probability mass covered by this instance
    """
    from .models import Models

    if models.is_empty():
        # Empty model, return zero
        return Polynomial(float_val=0.0), Polynomial(float_val=0.0)

    if hasattr(models, "factors"):
        Z = Polynomial(float_val=1.0)
        Z_covered = Polynomial(float_val=1.0)  # Covered mass that are not filtered

        for f in models.factors:
            if isinstance(f, Models):
                # Recurse; another Models instance
                f_Z, f_Z_covered = compute_Z(f)
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
                    f_Z = f_w_poly + Polynomial(float_val=1.0)

                    # Actually covered pmass
                    if f_coll is None:
                        f_Z_covered = f_Z
                    else:
                        if f_coll:
                            f_Z_covered = f_w_poly
                        else:
                            f_Z_covered = Polynomial(float_val=1.0)
                else:
                    # None-weight signifies 'absolute' (typically definitionally
                    # derived) facts, which do not even allow possibility of the
                    # literal not holding
                    f_Z = Polynomial(float_val=1.0)
                    if f_coll is None:
                        f_Z_covered = f_Z
                    else:
                        if f_coll:
                            f_Z_covered = f_Z
                        else:
                            f_Z_covered = Polynomial(float_val=0.0)

            if f_Z is not None:
                Z = Z * f_Z
                Z_covered = Z_covered * f_Z_covered
        
        return Z, Z_covered

    if hasattr(models, "outcomes"):
        bottom_models, branch_consequences = models.outcomes

        if branch_consequences is None:
            return compute_Z(bottom_models)
        else:
            Z = Polynomial(float_val=0.0)
            Z_covered = Polynomial(float_val=0.0)
            for bc in branch_consequences:
                # Each outcome entry is a tuple of a model for some program bottom in
                # the splitting sequence of the program (tree actually), a weight sum
                # applied to the bottom model, a Models instance for some program top,
                # and a boolean flag denoting whether this outcome branch is filtered
                # out and thus shouldn't contribute to 'covered' probability mass
                _, _, branch_w, top_base_w, top_models, filtered_out = bc

                assert isinstance(top_models, Models)
                top_Z, top_Z_covered = compute_Z(top_models)

                o_Z = branch_w * top_base_w * top_Z
                o_Z_covered = branch_w * top_base_w * top_Z_covered

                Z = Z + o_Z
                if not filtered_out:
                    Z_covered = Z_covered + o_Z_covered
            
            return Z, Z_covered

    raise ValueError("Invalid Models instance")
