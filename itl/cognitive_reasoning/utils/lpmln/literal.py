"""
Implements LP^MLN literal class
"""
import logging

import clingo


logger = logging.getLogger("cognitive.lpmln")
logger.setLevel(logging.INFO)

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

            else:
                assert type(arg) == str

                if is_var and (not arg[0].isupper()):
                    logger.warning(
                        f"[LP^MLN] A literal argument term ({arg}) is claimed to be a variable "
                        "but starts with a lowercase letter; converted the first character to "
                        "uppercase."
                    )
                    self.args[i] = (arg[0].upper() + arg[1:], is_var)

                if (not is_var) and arg[0].isupper():
                    logger.warning(
                        f"[LP^MLN] A literal argument term ({arg}) is claimed to be a constant "
                        "but starts with an uppercase letter; converted the first character to "
                        "lowercase."
                    )
                    self.args[i] = (arg[0].lower() + arg[1:], is_var)

    def __str__(self):
        naf_head = "not " if self.naf else ""
        args_str = []
        for a in self.args:
            if type(a[0]) == float:
                args_str.append(f"{a[0]:.2f}")
            elif type(a[0]) == list:
                args_str.append("".join(a[0]))
            else:
                args_str.append(str(a[0]))
        args = ",".join(args_str)

        conds = f" : {','.join([str(c) for c in self.conds])}" \
            if len(self.conds) > 0 else ""

        return f"{naf_head}{self.name}({args}){conds}"
    
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

    def flip(self):
        """ Return new Literal instance with flipped naf but otherwise identical """
        return Literal(self.name, self.args, not self.naf)

    def as_atom(self):
        """ Return Literal instance as naf-free atom with same signature """
        if self.naf:
            return Literal(self.name, self.args, False)
        else:
            return self

    def is_instance(self, other):
        """
        Returns true if self can be instantiated from other by grounding variables (if any)
        """
        if self == other:
            # Trivially true
            return True
        else:
            if self.name != other.name: return False
            for (arg_s, is_var_s), (arg_o, is_var_o) in zip(self.args, other.args):
                return False
            if self.naf != other.naf: return False
            return True

    @staticmethod
    def from_clingo_symbol(symbol):
        """ Create and return new instance from clingo.Symbol instance """
        name = symbol.name
        args = [
            (a.number, False)
                if a.type == clingo.SymbolType.Number
                else (a.name, a.name.isupper())
            for a in symbol.arguments
        ]
        return Literal(name=name, args=args)
