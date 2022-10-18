from ..polynomial import Polynomial
from ..utils import cacheable


class ModelsBranchOutcomes:
    """
    Representation of a set of branching events (with respect to some common factor
    sets representing the basis of a model collection), and additional consequences
    entailed by the particular branch event. Pair with a partner ModelsCommonFactors
    instance to form a complete model collection for a single-top-node program.
    """
    def __init__(self, outcomes=None):
        if outcomes is None:
            # Empty outcomes would represent consideration of zero branches originating
            # from its bottom factors counterpart, effectively disregarding all of them
            # and leading to zero models covered
            self.outcomes = []
        else:
            # Each branch outcome entry consists of:
            #   1) a set of literals specifying this particular branch
            #   2) the pairing ModelsCommonFactors filtered with the event spec
            #   3) ... and corresponding branch base weight
            #   4) top base weight possibly obtained from reducing some top program
            #      with the branch event
            #   5) ... and Models instance for the reduced program
            #   6) finally a boolean flag denoting whether this outcome branch is
            #      filtered out and thus shouldn't contribute to 'covered' models
            self.outcomes = outcomes
        
        self.cache = {
            method: {} for method in dir(self)
            if callable(getattr(self, method)) and not method.startswith("__")
        }

    def __repr__(self):
        return f"ModelsBranchOutcomes(len={len(self.outcomes)})"

    def is_empty(self):
        """
        Instance is empty if its list of outcomes is empty
        """
        return len(self.outcomes) == 0

    @cacheable
    def atoms(self):
        """ Return all atoms occurring covered by instance """
        atoms = set()
        for branch_outcome in self.outcomes:
            _, bf_filtered, _, _, top_models, filtered_out \
                = branch_outcome
            
            if filtered_out:
                # Disregard this outcome branch
                continue

            atoms |= bf_filtered.atoms() if bf_filtered is not None else set()
            atoms |= top_models.atoms() if top_models is not None else set()
        
        return atoms

    def enumerate(self):
        """
        Unroll the instance to generate every single model covered, along with its
        total weight
        """
        for branch_outcome in self.outcomes:
            branch_ev, _, branch_w, top_base_w, \
                top_models, filtered_out = branch_outcome

            if filtered_out:
                # Suppress branch weight to zero so that they are properly excluded
                # when computing Z, but don't ignore since all branches are needed
                # for finding universe of bottom branch event atoms
                branch_w = Polynomial(float_val=0.0)

            branch_atms = frozenset([lit for lit in branch_ev if lit.naf==False])

            if top_models is not None:
                for top_atms, top_model_w in top_models.enumerate():
                    yield ((branch_atms, branch_w), (top_atms, top_base_w*top_model_w))
            else:
                yield ((branch_atms, branch_w), (frozenset(), Polynomial(float_val=1.0)))

    @cacheable
    def compute_Z(self):
        """
        Compute and return total unnormalized probability mass covered by this instance
        """
        Z = Polynomial(float_val=0.0)
        for branch_outcome in self.outcomes:
            _, _, branch_w, top_base_w, \
                top_models, filtered_out = branch_outcome

            if filtered_out:
                # Disregard this outcome branch
                continue

            o_Z = branch_w * top_base_w

            if top_models is not None:
                o_Z = o_Z * top_models.compute_Z()

            Z = Z + o_Z

        return Z

    @cacheable
    def filter(self, literals):
        """
        Filter the instance to return a new instance representing the subset of
        models satisfying the condition specified by the *disjunction* of literals,
        so that formulas in conjunctive normal form can be processed with a chain of
        calls to filter.
        """
        # Filter each outcome and add to new outcome list if result is not empty
        filtered_outcomes = []
        for branch_outcome in self.outcomes:
            branch_ev, bf_filtered, branch_w, top_base_w, \
                top_models, filtered_out = branch_outcome

            if filtered_out:
                # This outcome branch is already filtered out, can add as-is right
                # away
                filtered_outcomes.append(branch_outcome)
                continue

            # Set of atoms actually covered by the outcome
            branch_atoms = bf_filtered.atoms()
            top_atoms = top_models.atoms()
            bc_atoms = branch_atoms | top_atoms

            # Disjunction to satisfy; making a copy that we can manipulate within this
            # for loop
            disjunction = {l for l in literals}

            add_as_is = False
            for l in literals:
                # literal shouldn't be covered by both branch_atoms and top_atoms
                assert not ((l.as_atom() in branch_atoms) and (l.as_atom() in top_atoms))

                if l.naf:
                    ## l.naf==True; l is negative literal
                    # If not covered by either branch_atoms or top_atoms, the disjunction
                    # is trivially satisfiable and this outcome can be included as-is without
                    # filtering
                    if l.as_atom() not in bc_atoms:
                        add_as_is = True

                    # If covered directly by branch_ev, this literal is never satisfiable and
                    # can be removed from disjunction
                    if l.as_atom() in branch_ev:
                        disjunction.remove(l)

                else:
                    ## l.naf==False; l is positive literal
                    # If covered directly by branch_ev, the disjunction is trivially satisfiable
                    # and this outcome can be included as-is without filtering
                    if l in branch_ev:
                        add_as_is = True

                    # If not covered by either branch_atoms or top_atoms, this literal is
                    # never satisfiable and can be removed from disjunction
                    if l not in bc_atoms:
                        disjunction.remove(l)

            if add_as_is:
                # No filtering needed, add the outcome as-is
                filtered_outcomes.append(branch_outcome)
                continue
            if len(disjunction) == 0:
                # Empty disjunction cannot be satisfied; flag as filtered and add to
                # outcome list
                filtered_outcomes.append(
                    (branch_ev, bf_filtered, branch_w, top_base_w, top_models, True)
                )
                continue

            # Outcomes that reached here represent mixture of models satisfying the
            # disjunction and those that do not; need further filtering of bf_filtered
            # and top_models, depending on how literals are covered by each
            literals_bm_f = frozenset([l for l in literals if l.as_atom() in branch_atoms])
            literals_top = frozenset([l for l in literals if l.as_atom() in top_atoms])

            if len(literals_bm_f) > 0:
                bm_further_filtered = bf_filtered.filter(literals_bm_f)
                branch_flt_w = bm_further_filtered.compute_Z()
            else:
                bm_further_filtered = bf_filtered
                branch_flt_w = branch_w

            if len(literals_top) > 0:
                top_models_filtered = top_models.filter(literals_top)
            else:
                top_models_filtered = top_models

            filtered_outcomes.append(
                (
                    branch_ev, bm_further_filtered, branch_flt_w,
                    top_base_w, top_models_filtered, filtered_out
                )
            )

        return ModelsBranchOutcomes(filtered_outcomes)
