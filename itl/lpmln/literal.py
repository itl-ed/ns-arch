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

    def as_atom(self):
        """ Return the naf-free atom version with same signature as new Literal instance """
        if self.naf:
            return Literal(self.name, self.args, False)
        else:
            return self
    
    def substitute(self, arg, new_arg, new_arg_is_var):
        """
        Return new Literal instance where all occurrences of arg are replaced with new_arg
        """
        new_args = [(new_arg, new_arg_is_var) if a[0] == arg else a for a in self.args]
        return Literal(self.name, new_args, self.naf)

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
