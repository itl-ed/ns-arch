"""
Implements LP^MLN weighted rule class
"""
from collections import defaultdict, Counter


class Rule:
    """ ASP rule, as comprehensible by popular ASP solvers like clingo. """
    def __init__(self, head=None, body=None, lb=None, ub=None):
        # Accept singletons
        if head is not None and not type(head) == list:
            head = [head]
        if body is not None and not type(body) == list:
            body = [body]

        self.head = [] if head is None else head
        self.body = [] if body is None else body

        # Lower bounds and upper bounds, used in case the rule is treated as choice rule
        self.lb = lb
        self.ub = ub
        if self.lb is not None and self.ub is not None:
            assert self.lb <= self.ub, "Lower bound should not be greater than upper bound"

    def __str__(self):
        if len(self.head) > 0 and (len(self.head) > 1 or len(self.head[0].conds) > 0):
            return self.str_as_choice()

        head_str = (str(self.head[0]) + (" " if len(self.body) else "")) \
            if self.head else ""
        body_str = \
            (":- " if len(self.body) else "") + \
            ", ".join([str(l) for l in self.body])

        return f"{head_str}{body_str}."
    
    def __repr__(self):
        return f"Rule({str(self)})"
    
    def __eq__(self, other):
        return \
            (self.head == other.head) and \
            (self.body == other.body) and \
            (self.lb == other.lb) and \
            (self.ub == other.ub)
    
    def __hash__(self):
        return hash(str(self))
    
    def flip(self):
        """
        Return flipped version as new rule, cast as integrity constraint.

        (Expected to be called when translating rule with weight "-a" (negative inf;
        i.e. zero-probability) into pure ASP rule)
        """
        assert len(self.head) > 0, "Cannot flip integrity constraint"

        return Rule(body=self.body+self.head)
    
    def str_as_choice(self):
        """ As string, treated as choice rule """
        assert len(self.head) > 0, "Cannot cast integrity constraint as choice rule"

        lb = "" if self.lb is None else f"{self.lb} "
        ub = "" if self.ub is None else f" {self.ub}"

        head_str = f"{lb}{{ {'; '.join([str(hl) for hl in self.head])} }}{ub}" + \
            (" " if len(self.body) else "")
        body_str = \
            (":- " if len(self.body) else "") + \
            ", ".join([str(l) for l in self.body])
            
        return f"{head_str}{body_str}."
    
    def body_contains(self, literal):
        contained = False
        for bl in self.body:
            if bl == literal:
                contained = True; break
        return contained

    def terms(self):
        """ Return set of all terms (constant or variable) occurring in the rule """
        terms_head = set.union(*[set(hl.args) for hl in self.head]) \
            if len(self.head) else set()
        terms_body = set.union(*[set(bl.args) for bl in self.body]) \
            if len(self.body) else set()
        return terms_head | terms_body

    def literals(self):
        """ Return set of all atoms occurring in the rule """
        return set(self.head + self.body)

    def is_fact(self):
        """ Return True if rule consists of its head and empty body """
        return (len(self.head) == 1) and (len(self.body) == 0)

    def is_grounded(self):
        """ Returns True if the rule is variable-free """
        return all([l.is_grounded() for l in self.literals()])

    def is_instance(self, other):
        """
        Returns true if self can be instantiated from other by grounding variables (if any)
        """
        # False if heads/bodies have different length
        if len(self.head) != len(other.head): return False
        if len(self.body) != len(other.body): return False

        terms_s = self.terms(); terms_o = other.terms()

        consts_s = {t for t, is_var in terms_s if not is_var}
        consts_o = {t for t, is_var in terms_o if not is_var}

        for c_o in consts_o:
            if c_o not in consts_s:
                return False            # Found constant that cannot be matched

        # Now trying unification for those that passed tests so far
        vars_o = [t for t, is_var in terms_o if is_var]
        consts_s = {t for t, is_var in terms_s if not is_var}

        # Obtain signatures for each term occurring in rule
        def term_signatures(rule):
            sigs = defaultdict(list)
            for hl in rule.head:
                for i, a in enumerate(hl.args):
                    sigs[a].append((hl.name, hl.naf, i))
            for bl in rule.body:
                for i, a in enumerate(bl.args):
                    sigs[a].append((bl.name, bl.naf, i))
            return dict(sigs)
        
        sigs_s = term_signatures(self); sigs_o = term_signatures(other)

        # First unify constants
        for c_o in consts_o:
            if (c_o, False) in sigs_s:
                for sig in sigs_o[(c_o, False)]:
                    if sig in sigs_s[(c_o, False)]:
                        # Clear common signatures
                        sigs_s[(c_o, False)].remove(sig)
                        if len(sigs_s[(c_o, False)]) == 0: del sigs_s[(c_o, False)]
                    else:
                        # The constants cannot unify
                        return False

        # Try variable substitution; keep looking for substitutable constant to
        # 'consume' until vars_o is exhausted. If feasible, must succeed in whichever
        # order.
        while len(vars_o) > 0:
            v = vars_o.pop()

            for t_s, sig in list(sigs_s.items()):
                cnt1 = Counter(sigs_o[(v, True)])
                cnt2 = Counter(sig)

                if cnt1 & cnt2 == cnt1:
                    # sigs_o[v] can unify with sig
                    for s in sigs_o[v]: sig.remove(s)
                    if len(sig) == 0: del sigs_s[t_s]
                    break
            else:
                # Cannot unify with any constant
                return False

        assert len(sigs_s) == 0, "Unification failure?"

        return True

    def substitute(self, val, new_val, is_pred):
        """
        Return new Rule instance where all occurrences of designated arg or pred are
        replaced with provided new value
        """
        new_head = [hl.substitute(val, new_val, is_pred) for hl in self.head]
        new_body = [bl.substitute(val, new_val, is_pred) for bl in self.body]

        return Rule(head=new_head, body=new_body)
