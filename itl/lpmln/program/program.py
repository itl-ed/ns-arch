"""
Implements LP^MLN program class
"""
from collections import defaultdict

from .solve import recursive_solve
from .optimize import optimize
from .split import split_program
from .reduce import reduce_program
from ..literal import Literal
from ..rule import Rule
from ..utils import logit


LARGE = 2e1           # Sufficiently large logit to use in place of, say, float('inf')
SCALE_PREC = 3e2      # For preserving some float weight precision
TOPK_RATIO = 0.75     # Percentage of answer sets to cover, by probability mass

class Program:
    """ Probabilistic ASP program, implemented as a list of weighted ASP rules. """
    def __init__(self, rules=None):
        self.rules = [] if rules is None else rules

        self._rules_by_atom = defaultdict(set)      
        for i, (rule, _) in enumerate(self.rules):
            for hl in rule.head:
                self._rules_by_atom[hl.as_atom()].add(i)
            for bl in rule.body:
                self._rules_by_atom[bl.as_atom()].add(i)

    def __len__(self):
        return len(self.rules)

    def __str__(self):
        """ LP^MLN program string representation """
        prog_s = ""

        weight_strs = []; max_ws_len = 0
        for _, r_pr in self.rules:
            if r_pr is None:
                weight_strs.append("A")
            else:
                if len(r_pr) == 1:
                    r_pr_str = f"logit({r_pr[0]:.3f})"
                else:
                    r_pr_str = ",".join([f"logit({p:.3f})" for p in r_pr])
                    r_pr_str = f"[{r_pr_str}]"
                weight_strs.append(r_pr_str)
            
            max_ws_len = max(len(weight_strs[-1]), max_ws_len)

        for (rule, _), ws in zip(self.rules, weight_strs):
            ws = ws + (" " * (max_ws_len-len(ws)))
            prog_s += f"{ws} ::   {str(rule)}\n"

        return prog_s

    def __repr__(self):
        return f"Program(len={len(self)})"

    def __add__(self, other):
        assert isinstance(other, Program), "Added value must be a Program"

        return Program(self.rules + other.rules)
    
    def __iadd__(self, other):
        return self + other
    
    def add_rule(self, rule, r_pr=None):
        """
        Probability value of 0 or 1 indicates hard-weighted rules. If r_pr is not provided,
        assume value(s) of 0.5 (effectively giving zero weights)
        """
        if len(rule.head) > 0:
            if r_pr is None:
                r_pr = [0.5] * len(rule.head)
            if type(r_pr) != list:
                r_pr = [r_pr] * len(rule.head)
        else:
            r_pr = [r_pr]
        r_pr = tuple(r_pr)

        assert isinstance(rule, Rule), "Added value must be a Rule"
        for p in r_pr:
            assert 0 <= p <= 1, "Must provide valid probability value to compute rule weight"

        self.rules.append((rule, r_pr))

        for hl in rule.head:
            self._rules_by_atom[hl.as_atom()].add(len(self.rules)-1)
        for bl in rule.body:
            self._rules_by_atom[bl.as_atom()].add(len(self.rules)-1)
    
    def add_absolute_rule(self, rule):
        """
        Add as an 'absolute' rule; apart from all the LP^MLN and probabilistic
        goodness, these are rather 'definitional' rules such that there's really
        no point in suspecting them they might not hold after all
        """
        self.rules.append((rule, None))

        for hl in rule.head:
            self._rules_by_atom[hl.as_atom()].add(len(self.rules)-1)
        for bl in rule.body:
            self._rules_by_atom[bl.as_atom()].add(len(self.rules)-1)

    def solve(self, topk_ratio=TOPK_RATIO):
        """ Wraps around recursive_solve, and exposed as class instance method """
        return recursive_solve(self, scale_prec=SCALE_PREC)

    def optimize(self, statements):
        """
        Solve for the optimal model as specified by [statements] argument. Optimal solution
        is found by composing appropriate optimization statements in clingo, which is attached
        to string representation of the program, and solving the program with clingo.
        
        Each optimization statement designates how much weight each occurrence of relevant
        literals (counted at most once for tuple) should contribute to the total weight.
        Statements that come earlier in the list will have higher priority.
        """
        return optimize(self, statements)
        
    def reduce(self, atoms):
        """ Return the program obtained by reducing self with given values of atoms """
        return reduce_program(self, atoms)

    @staticmethod
    def split(comp, atoms_map, grounded_rules_rel):
        """
        Search for a minimal splitting set for comp and return the corresponding split
        programs.

        (Implements paper [[How to Split a Logic Program]] (Ben-Eliyahu-Zohary, 2021).)
        """
        return split_program(comp, atoms_map, grounded_rules_rel)

    def _pure_ASP_str(self, unsats=False):
        """
        Return string compilation into to pure ASP program string understandable by
        clingo.

        Feeding unsats=True adds 'unsat' atoms, which enables tracking of model weight
        sums as well as violated choice rules.
        
        Feeding unsats=False is primarily for the purpose of computing dependency
        graph of the grounded program, or for preparing vanilla ASP program (i.e. not
        a LP^MLN program) not related to any type of probabilistic inference.
        """
        as_str = ""

        for ri, (rule, r_pr) in enumerate(self.rules):
            if r_pr is None:
                # No point in casting these 'absolute' rules as choice rules,
                # simply add them as rule string as it stands
                as_str += str(rule) + "\n"
                continue

            if unsats:
                if len(rule.head) <= 1:
                    if len(rule.head) == 1:
                        # Add original rule as choice rule, as possibly deriving
                        # rule head when body holds
                        as_str += rule.str_as_choice() + "\n"

                    # Add rule that derives unsat atom with appropriate rule weight
                    # whenever rule body holds but none of rule head holds
                    weight = logit(r_pr[0], large="a")
                    if type(weight)!=str:
                        weight = int(weight * SCALE_PREC)

                    unsat_args = [(ri, False), (weight, False)]
                    unsat_rule = Rule(
                        head=Literal("unsat", unsat_args),
                        body=rule.body+[hl.flip() for hl in rule.head]
                    )
                    as_str += str(unsat_rule) + "\n"
                else:
                    raise NotImplementedError
            else:
                if len(rule.head) <= 1:
                    if r_pr[0] == 1.0:
                        # Add as-is
                        as_str += str(rule) + "\n"
                    elif r_pr[0] == 0.0:
                        # Turn into corresponding integrity constraint and add
                        as_str += "".join(str(nr)+"\n" for nr in rule.negate())
                    else:
                        if len(rule.head) == 1:
                            as_str += rule.str_as_choice() + "\n"
                        else:
                            # Even when unsats=False, should add a modified rule with
                            # an auxiliary atom as head when body contains more than
                            # one literal, so that body literals are connected in the
                            # dependency graph
                            if len(rule.body) > 1:
                                aux_args = [(ri, False)]
                                aux_rule = Rule(
                                    head=Literal("unsat", aux_args),
                                    body=rule.body
                                )
                                as_str += str(aux_rule) + "\n"
                else:
                    # Choice rules with multiple head literals
                    as_str += rule.str_as_choice() + "\n"

        return as_str
