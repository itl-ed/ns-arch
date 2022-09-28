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

    def nonfn_terms(self):
        """ Return set of all non-function argument terms occurring in the literal """
        terms = set()
        for a_val, a_is_var in self.args:
            if type(a_val)==tuple:
                terms |= {(fa, fa[0].isupper()) for fa in a_val[1]}
            else:
                terms.add((a_val, a_is_var))

        return terms

    def substitute(self, terms=None, functions=None, preds=None):
        """
        Return new Literal instance where all occurrences of designated arg, fn or pred
        are replaced with provided new value
        """
        terms_map = terms or {}
        term_names_map = { t1[0]: t2[0] for t1, t2 in terms_map.items() }
        fns_map = functions or {}
        preds_map = preds or {}

        term_is_var_map = { a[0]: a[1] for a in self.args if type(a[0])==str }
        term_is_var_map.update({ t2[0]: t2[1] for t2 in terms_map.values() })

        if self.name == "*_?" and self.args[0][0] in preds_map:
            # Substituting the reserved predicate "*_?" with a contentful predicate
            subs_name = preds_map[self.args[0][0]]
            self_args = self.args[1:]
        else:
            subs_name = preds_map.get(self.name, self.name)
            self_args = self.args

        subs_args = []
        for arg in self_args:
            a_term, _ = arg

            if type(a_term)==tuple:
                # Function term
                if arg in terms_map:
                    # The whole function term is to be subtituted
                    subs_args.append(terms_map.get(arg, arg))
                else:
                    # Appropriately substitute function name and args if applicable
                    f_name, f_args = a_term
                    subs_f_name = fns_map.get(f_name, f_name)
                    subs_f_args = tuple(term_names_map.get(fa, fa) for fa in f_args)
                    subs_is_var = any(
                        term_is_var_map.get(fa, fa[0].isupper()) for fa in subs_f_args
                    )
                    subs_args.append(
                        ((subs_f_name, subs_f_args), subs_is_var)
                    )
            else:
                # Non-function term; simple substitution
                subs_args.append(terms_map.get(arg, arg))

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

    @staticmethod
    def isomorphism_btw(lits1, lits2, ism):
        """
        Helper method for testing whether two iterables of literals are isomorphic up to
        variable & function renaming. Return an isomorphism if found to be so; otherwise,
        return None.
        """
        isomorphism = ism or {
            "terms": {}, "functions": {}
        }
        for hl_s in lits1:
            lit_matched = False

            for hl_o in lits2:
                potential_mapping = { "terms": {}, "functions": {} }
                args_mappable = True

                if hl_s.name != hl_o.name:
                    # Predicate name mismatch
                    continue

                for sa, oa in zip(hl_s.args, hl_o.args):
                    sa_term, sa_is_var = sa
                    oa_term, oa_is_var = oa

                    if sa_is_var != oa_is_var:
                        # Term type mismatch
                        args_mappable = False; break

                    if sa_is_var == oa_is_var == False:
                        if sa_term != oa_term:
                            # Constant term mismatch
                            args_mappable = False; break
                    else:
                        if type(sa_term) != type(oa_term):
                            # Function vs. non-function term mismatch
                            args_mappable = False; break

                        if type(sa_term) == type(oa_term) == str:
                            # Both args are variable terms
                            if sa_term in isomorphism["terms"]:
                                if isomorphism["terms"][sa_term] != oa_term:
                                    # Conflict with existing mapping
                                    args_mappable = False; break
                            elif sa_term in potential_mapping["terms"]:
                                if potential_mapping["terms"][sa_term] != oa_term:
                                    # Conflict with existing potential mapping
                                    args_mappable = False; break
                            else:
                                # Record potential mapping
                                potential_mapping["terms"][sa_term] = oa_term

                        elif type(sa_term) == type(oa_term) == tuple:
                            sa_f_name, sa_f_args = sa_term
                            oa_f_name, oa_f_args = oa_term

                            if len(sa_f_args) != len(oa_f_args):
                                # Function arity mismatch
                                args_mappable = False; break

                            # Function name mismatch
                            if sa_f_name in isomorphism["functions"]:
                                if isomorphism["functions"][sa_f_name] != oa_f_name:
                                    # Conflict with existing mapping
                                    args_mappable = False; break
                            elif sa_f_name in potential_mapping["functions"]:
                                if potential_mapping["functions"][sa_f_name] != oa_f_name:
                                    # Conflict with existing potential mapping
                                    args_mappable = False; break
                            else:
                                # Record potential mapping
                                potential_mapping["functions"][sa_f_name] = oa_f_name

                            for sfa, ofa in zip(sa_f_args, oa_f_args):
                                if sfa in isomorphism["terms"]:
                                    if isomorphism["terms"][sfa] != ofa:
                                        # Conflict with existing mapping
                                        args_mappable = False; break
                                elif sfa in potential_mapping["terms"]:
                                    if potential_mapping["terms"][sfa] != ofa:
                                        # Conflict with existing potential mapping
                                        args_mappable = False; break
                                else:
                                    # Both args are variable terms, record potential mapping
                                    potential_mapping["terms"][sfa] = ofa
                        
                        else:
                            raise NotImplementedError

                if args_mappable:
                    # Update isomorphism
                    lit_matched = True
                    isomorphism["terms"].update(potential_mapping["terms"])
                    isomorphism["functions"].update(potential_mapping["functions"])
                else:
                    # Discard potential mapping and move on
                    continue

            # Return None as soon as any literal is found to be unmappable to any
            if not lit_matched: return None

        # Successfully found an isomorphism
        return isomorphism
