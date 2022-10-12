""" Implements utility methods for manipulating logic formulas """
from itertools import product
from collections import defaultdict


def cnf_to_dnf(cnf):
    """
    Converts a logical formula already in conjunctive normal for (CNF) to
    disjunctive normal form (DNF)
    """
    # Applying distributive law, making it DNF
    dnf = list(product(*cnf))

    # Simplify and return
    return simplify_dnf(dnf)

def dnf_to_cnf(dnf):
    """
    Converts a logical formula already in disjunctive normal for (DNF) to
    conjunctive normal form (CNF)
    """
    # Applying distributive law, making it CNF
    cnf = list(product(*dnf))

    # Simplify and return
    return simplify_cnf(cnf)

def simplify_cnf(cnf):
    """
    Simplify a logical formula in CNF, by first simplifying each conjunct and
    then performing any possible cross-conjunct resolution
    """
    # Simplify each conjunct
    cnf_simplified = [simplify_disjunction(cjct) for cjct in cnf]

    # Remove conjuncts that are found to be tautologies
    cnf_simplified = [
        cjct for cjct in cnf_simplified if type(cjct)==set
    ]

    if len(cnf_simplified) <= 1:
        # Can return at this point without cross-conjunct checks
        return cnf_simplified

    # Any conjuncts that are supersets of some other can be removed
    supersets_removed = []
    for cjct in cnf_simplified:
        if not any(cjct >= cjct2 for cjct2 in supersets_removed):
            supersets_removed.append(cjct)
    cnf_simplified = supersets_removed

    resolution_finished = False
    while not resolution_finished:
        # All occurring atoms, and whether both themselves and their negations occur
        # in different conjuncts
        occurring_atoms = {lit.as_atom() for cjct in cnf_simplified for lit in cjct}
        occurring_atoms = {
            atm: (
                [cjct - {atm} for cjct in cnf_simplified
                    if atm in cjct],                        # Positive occurrences
                [cjct - {atm.flip()} for cjct in cnf_simplified
                    if atm.flip() in cjct],                 # Negative occurrences
                [cjct for cjct in cnf_simplified
                    if not (atm in cjct or atm.flip() in cjct)],    # Irrelevant
            )
            for atm in occurring_atoms
        }
        occurring_atoms = {
            atm: (simplify_cnf(pos_occs), simplify_cnf(neg_occs), irrel_cjcts)
            for atm, (pos_occs, neg_occs, irrel_cjcts) in occurring_atoms.items()
        }

        resolution_possible = {
            atm for atm, (pos_occs, neg_occs, _) in occurring_atoms.items()
            if len(pos_occs) > 0 and len(neg_occs) > 0
        }

        if len(resolution_possible) == 0:
            # No resolution possible, return
            resolution_finished = True
        else:
            for lit in resolution_possible:
                pos_occs, neg_occs, irrel_cjcts = occurring_atoms[lit]

                # Propositional resolution (p or Q) and (~p or R) => (Q or R)
                resolution_result = [
                    p_cjct | n_cjct
                    for p_cjct, n_cjct in product(pos_occs, neg_occs)
                ]
                cnf_simplified = resolution_result + irrel_cjcts
                
                break       # Processing one literal at a time...

    return cnf_simplified

def simplify_dnf(dnf):
    """
    Simplify a logical formula in DNF
    """
    # Simplify each disjunct
    dnf_simplified = [simplify_conjunction(djct) for djct in dnf]

    # Remove disjuncts that are found to be contraditions
    dnf_simplified = [
        djct for djct in dnf_simplified if type(djct)==set
    ]

    if len(dnf_simplified) <= 1:
        # Can return at this point without cross-disjunct checks
        return dnf_simplified
    
    # Any disjuncts that are subsets of some other can be removed
    subsets_removed = []
    for djct in dnf_simplified:
        if not any(djct >= djct2 for djct2 in subsets_removed):
            subsets_removed.append(djct)
    dnf_simplified = subsets_removed

    return dnf_simplified

def simplify_disjunction(disj):
    """
    Simplify a flat disjunction of literals, by processing any duplicate literals
    within the disjunction (positive or negative)
    """
    if len(disj) == 0: return disj      # Empty; return as-is

    disj_simplified = defaultdict(set)
    for d_lit in disj:
        disj_simplified[d_lit.as_atom()].add(d_lit)
    
    if any(len(d_lits)==2 for d_lits in disj_simplified.values()):
        # Disjunction contains (P or ~P) for some atom P, which is a tautalogy and
        # thus collapses the whole disjunction into True
        return True
    else:
        assert all(len(d_lits)==1 for d_lits in disj_simplified.values())
        return set.union(*disj_simplified.values())

def simplify_conjunction(conj):
    """
    Simplify a flat conjunction of literals, by processing any duplicate literals
    within the conjunction (positive or negative)
    """
    if len(conj) == 0: return conj      # Empty; return as-is

    conj_simplified = defaultdict(set)
    for c_lit in conj:
        conj_simplified[c_lit.as_atom()].add(c_lit)
    
    if any(len(c_lits)==2 for c_lits in conj_simplified.values()):
        # Conjunction contains (P and ~P) for some atom P, which is a contradiction
        # and thus collapses the whole conjunction into False
        return False
    else:
        assert all(len(c_lits)==1 for c_lits in conj_simplified.values())
        return set.union(*conj_simplified.values())
