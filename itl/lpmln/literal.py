"""
Implements LP^MLN literal class
"""
import clingo


class Literal:
    """
    ASP literal, as comprehensible by popular ASP solvers like clingo. We do not
    allow function terms that are not null-ary as arguments, at least for now.
    """
    def __init__(self, name, args=None, naf=False, conds=None):
        self.name = name
        self.args = [] if args is None else args
        self.naf = naf         # Soft negative, or 'negation-as-failure'
        self.conds = [] if conds is None else conds
                               # Condition literals, as meant in clingo conditions

        for i in range(len(self.args)):
            arg, is_var = self.args[i]

            if type(arg) == float or type(arg) == int:
                # Do not allow number arguments as variables
                if is_var:
                    raise ValueError("Number argument is claimed to be a variable")
            
            elif type(arg) == list:
                # List argument is a chain of variables and operators forming a (hopefully) valid
                # arithmetic formula
                ...

            elif type(arg) == tuple:
                # Uninterpreted function term
                if is_var != any([a[0].isupper() for a in arg[1]]):
                    raise ValueError("Term letter case and variable claim mismatch")

            else:
                assert type(arg) == str

                if is_var != arg[0].isupper():
                    raise ValueError("Term letter case and variable claim mismatch")

    def __str__(self):
        naf_head = "not " if self.naf else ""
        args_str = []
        for a in self.args:
            if type(a[0]) == float:
                args_str.append(f"{a[0]:.2f}")
            elif type(a[0]) == list:
                args_str.append("".join(a[0]))
            elif type(a[0]) == tuple:
                args_str.append(f"{a[0][0]}({','.join(a[0][1])})")
            else:
                args_str.append(str(a[0]))
        args = "("+",".join(args_str)+")" if len(args_str)>0 else ""

        conds = f" : {','.join([str(c) for c in self.conds])}" \
            if len(self.conds) > 0 else ""

        return f"{naf_head}{self.name}{args}{conds}"
    
    def __repr__(self):
        return f"Literal({str(self)})"
    
    def __eq__(self, other):
        """ Literal equality comparison """
        return \
            (self.name == other.name) and \
            (self.args == other.args) and \
            (self.naf == other.naf) and \
            (self.conds == other.conds)
    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(str(self))

    def is_grounded(self):
        """ Returns True if the literal is variable-free """
        return not any([is_var for _, is_var in self.args])

    def flip(self):
        """ Return new Literal instance with flipped naf but otherwise identical """
        return Literal(self.name, self.args, not self.naf)
    
    def flip_classical(self):
        """
        Return new Literal instance with strong-negated predicate name but otherwise
        identical
        """
        if self.name.startswith("-"):
            # Already strong-negated; cancel negation
            return Literal(self.name[1:], self.args, self.naf)
        else:
            # Strong negation by attaching "-" (clingo style)
            return Literal("-"+self.name, self.args, self.naf)

    def as_atom(self):
        """ Return the naf-free atom version with same signature as new Literal instance """
        if self.naf:
            return Literal(self.name, self.args, False)
        else:
            return self
    
    def substitute(self, subs_map):
        """
        Return new Rule instance where all occurrences of designated arg or pred are
        replaced with provided new value
        """
        if self.name == "*_?" and self.args[0][0] in subs_map:
            # Substituting the reserved predicate "*_?" with a contentful predicate
            subs_name = subs_map[self.args[0][0]]
            self_args = self.args[1:]
        else:
            subs_name = subs_map.get(self.name, self.name)
            self_args = self.args

        subs_args = []
        for a_term, a_is_var in self_args:
            if type(a_term)==tuple:
                # Function term; appropriately substitute function name and args
                f_name, f_args = a_term
                subs_f_name = subs_map.get(f_name, f_name)
                subs_f_args = tuple(subs_map.get(fa, fa) for fa in f_args)
                subs_args.append(
                    ((subs_f_name, subs_f_args), a_is_var)
                )
            else:
                # Non-function term; simple substitution
                subs_args.append((subs_map.get(a_term, a_term), a_is_var))

        return Literal(subs_name, subs_args, self.naf)

    @staticmethod
    def from_clingo_symbol(symbol):
        """ Create and return new instance from clingo.Symbol instance """
        name = symbol.name
        args = [
            (a.number, False)
                if a.type == clingo.SymbolType.Number
                else ((a.name, a.name.isupper())                  # Constant
                    if len(a.arguments)==0
                    else ((a.name, tuple([t.name for t in a.arguments])), False)   # Function
                )
            for a in symbol.arguments
        ]
        return Literal(name=name, args=args)
