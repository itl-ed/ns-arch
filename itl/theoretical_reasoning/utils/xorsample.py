"""
Utility methods required for XORSample algorithm ((near)-uniform sampling of answer
sets). Basically pilfered from `xorro` package; couldn't use it directly as it is
not forward-compatible with clingo 5.5+...
"""
import math
import random
import itertools

import clingo


PARS = ["odd", "even"]


def xorsample(prog, k=1, s=None, q=0.5):
    xors = _generate_random_xors(prog, s, q)
    xprog = _xors_as_program(xors)

    ctl = clingo.Control()
    ctl.add("base", [], prog+xprog)
    ctl.ground([("base", [])])
    ctl.configuration.solve.models = k

    # Part corresponding to xorro.translate(mode=="tree")
    ret = _symbols_to_xor_r(ctl.symbolic_atoms)
    with ctl.backend() as backend:
        if ret is None:
            backend.add_rule([], [])
        else:
            constraints, facts = ret
            for fact in facts:
                backend.add_rule([], [-fact])
            for constraint in constraints:
                tree = _to_tree(constraint)
                backend.add_rule([], [-tree.translate(backend)])
    
    models = []
    with ctl.solve(yield_=True) as solve_gen:
        for m in solve_gen:
            atoms = m.symbols(atoms=True)
            models.append(
                (atoms, sum([a.arguments[1].number for a in atoms if a.name=="unsat"]))
            )
        
            if solve_gen.get().unsatisfiable: break

    return models


def _generate_random_xors(prog, s, q):
    """
    Sample parity constraints concerning symbolic atoms in a grounded clingo.Control
    instance
    """
    ctl = clingo.Control()
    ctl.add("base", [], prog)
    ctl.ground([("base", [])])

    symbols = [
        atom.symbol for atom in ctl.symbolic_atoms
        if (not atom.is_fact) and (atom.symbol.name != "unsat")
    ]

    # If not provided, use this heuristic for the number of parity constraints to sample
    if s is None:
        s = int(math.log(len(symbols) + 1, 2))

    return [
        (random.sample(symbols, int(len(symbols)*q)), random.randint(0,1))
        for _ in range(s)       # Sampled constraint body atoms & parity
    ]


def _xors_as_program(xors):
    """Encode given set of parity constraints as a clingo program block"""
    prog = ""

    for c, xor in enumerate(xors):
        xor_body, xor_parity = xor
        prog += f"__parity({c},{PARS[xor_parity]}).\n"

        for a in xor_body:
            prog += f"__parity({c},{PARS[xor_parity]},({a},)):-{a}.\n"

    return prog


def _symbols_to_xor_r(symbolic_atoms):
    constraints = {}
    lits = []

    for atom in symbolic_atoms.by_signature("__parity",2):
        cid = atom.symbol.arguments[0].number
        par = atom.symbol.arguments[1].name
        constraints[cid] = _XORConstraint(str(par) == "odd")
    
    for atom in symbolic_atoms.by_signature("__parity",3):
        constraint = constraints[atom.symbol.arguments[0].number]
        lit = atom.literal
        truth = True if atom.is_fact else None

        if truth:
            constraint.parity = not constraint.parity
        elif truth is None:
            if lit in constraint.literals:
                constraint.literals.remove(lit)
            elif -lit in constraint.literals:
                constraint.literals.remove(-lit)
                constraint.parity = not constraint.parity
            else:
                if lit < 0:
                    constraint.literals.add(abs(lit))
                    constraint.parity = not constraint.parity
                else:
                    constraint.literals.add(lit)
                if abs(lit) not in lits:
                    lits.append(abs(lit))
    
    facts = set()
    result = []
    for constraint in constraints.values():
        literals = sorted(constraint.literals)
        n = len(literals)
        if n == 0:
            if constraint.parity == 1:
                return None
        else:
            if constraint.parity == 0:
                literals[0] = -literals[0]
            if n > 1:
                result.append(literals)
            else:
                facts.add(literals[0])

    return result, sorted(facts)


def _to_tree(constraint):
    layer = [_Leaf(literal) for literal in constraint]
    tree = lambda l, r: l if r is None else _Tree(l, r)

    while len(layer) > 1:
        layer = itertools.starmap(
            tree, itertools.zip_longest(layer[0::2], layer[1::2])
        )
        layer = list(layer)

    return layer[0]


class _XORConstraint:
    def __init__(self, parity):
        self.parity = parity
        self.literals = set()


class _Leaf:
    def __init__(self, atom):
        self._atom = atom
    def translate(self, backend):
        return self._atom

class _Tree:
    def __init__(self, lhs, rhs):
        self._lhs = lhs
        self._rhs = rhs
    def translate(self, backend):
        lhs = self._lhs.translate(backend)
        rhs = self._rhs.translate(backend)

        aux = backend.add_atom()
        backend.add_rule([aux], [ lhs, -rhs])
        backend.add_rule([aux], [-lhs,  rhs])

        return aux
