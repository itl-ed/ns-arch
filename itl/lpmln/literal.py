"""
Implements LP^MLN literal class
"""
from itertools import permutations

import clingo

from .utils import unify_mappings


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
            (isinstance(self, Literal) and isinstance(other, Literal)) and \
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
    def entailing_mapping_btw(iter1, iter2, mapping=None):
        """
        Helper method for testing whether two finite iterables of literals or negated
        conjunctions of literals, representing conjunctions of the components, are
        (partially) mappable up to variable & function renaming, so that one entails
        the other under the mapping. Return a mapping if found one, along with 3-valued
        flag of the direction of entailment.
        """
        mapping = mapping or { "terms": {}, "functions": {} }

        # Cast into sequences to assign indicies, in order to keep track of
        # entailability between components
        iter1 = tuple(iter1); iter2 = tuple(iter2)
        entailabilities = {}

        # 3-valued flags indicate value of iter1 - iter2 (analogously...):
        #   0: iter1 and iter2 are equivalent (iter1 == iter2)
        #   1: iter1 is strictly stronger and entails iter2 (iter1 > iter2)
        #   -1: iter2 is strictly stronger and entails iter1 (iter1 < iter2)

        for i1, cnjt1 in enumerate(iter1):
            # Note: list type encodes negated conjunction
            assert isinstance(cnjt1, Literal) or isinstance(cnjt1, list)

            for i2, cnjt2 in enumerate(iter2):
                assert isinstance(cnjt2, Literal) or isinstance(cnjt2, list)

                if isinstance(cnjt1, Literal) and isinstance(cnjt2, Literal):
                    # Base case, both cnjt1 and cnjt2 are literals
                    potential_mapping = { "terms": {}, "functions": {} }
                    args_mappable = True

                    if cnjt1.name != cnjt2.name:
                        # Predicate name mismatch
                        entailabilities[(i1, i2)] = (None, None)
                        continue

                    for sa, oa in zip(cnjt1.args, cnjt2.args):
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
                                if sa_term in mapping["terms"]:
                                    if mapping["terms"][sa_term] != oa_term:
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
                                if sa_f_name in mapping["functions"]:
                                    if mapping["functions"][sa_f_name] != oa_f_name:
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
                                    if sfa in mapping["terms"]:
                                        if mapping["terms"][sfa] != ofa:
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
                        # Potential mapping found
                        entailabilities[(i1, i2)] = (potential_mapping, 0)
                    else:
                        # Not entailable; discard potential mapping and move on
                        entailabilities[(i1, i2)] = (None, None)

                elif isinstance(cnjt1, list) and isinstance(cnjt2, list):
                    # Recursive call for trying to find a mapping between the
                    # two negated conjunctions
                    nc_mapping, nc_entail_dir = Literal.entailing_mapping_btw(
                        cnjt1, cnjt2, mapping
                    )
                    entailabilities[(i1, i2)] = (nc_mapping, nc_entail_dir)

                else:
                    # Type mismatch or invalid types
                    entailabilities[(i1, i2)] = (None, None)

        # In order to find a valid final mapping, the mappings between components
        # in iter1 & iter2 should be unifiable, either literals in iter1 or iter2
        # must be exhausted ahd have consistent entailment directions. A mapping
        # is considered 'complete' if both are exhausted (same length).
        valid_prms = []
        if len(iter1) >= len(iter2):
            # All conjuncts in iter2 must be matched to some conjunct in iter1
            for prm in permutations(range(len(iter1)), len(iter2)):
                to_unify = [
                    entailabilities[(i1, i2)] for i2, i1 in enumerate(prm)
                ]
                all_matched = all(
                    mapping_piece is not None for mapping_piece, _ in to_unify
                )
                directions_consistent = not {1, -1}.issubset(
                    {ent_dir for _, ent_dir in to_unify}
                )
                if all_matched and directions_consistent:
                    terms_unified = unify_mappings(
                        [mapping["terms"]] + \
                            [mapping_piece["terms"] for mapping_piece, _ in to_unify]
                    )
                    fns_unified = unify_mappings(
                        [mapping["functions"]] + \
                            [mapping_piece["functions"] for mapping_piece, _ in to_unify]
                    )
                    if terms_unified is not None and fns_unified is not None:
                        valid_prms.append((terms_unified, fns_unified))
        else:
            # len(iter1) < len(iter2):
            # All conjuncts in iter1 must be matched to some conjunct in iter2
            for prm in permutations(range(len(iter2)), len(iter1)):
                to_unify = [
                    entailabilities[(i1, i2)] for i1, i2 in enumerate(prm)
                ]
                all_matched = all(
                    mapping_piece is not None for mapping_piece, _ in to_unify
                )
                directions_consistent = not {1, -1}.issubset(
                    {ent_dir for _, ent_dir in to_unify}
                )
                if all_matched and directions_consistent:
                    terms_unified = unify_mappings(
                        [mapping["terms"]] + \
                            [mapping_piece["terms"] for mapping_piece, _ in to_unify]
                    )
                    fns_unified = unify_mappings(
                        [mapping["functions"]] + \
                            [mapping_piece["functions"] for mapping_piece, _ in to_unify]
                    )
                    if terms_unified is not None and fns_unified is not None:
                        valid_prms.append((terms_unified, fns_unified))
        
        if len(valid_prms) > 0:
            # Valid permutation(s) found; use the first one in the list (not sure
            # if there's any real difference between prms when more than one found)
            final_mapping = {
                "terms": valid_prms[0][0], "functions": valid_prms[0][0]
            }
            if len(iter1) > len(iter2):
                entail_dir = 1
            elif len(iter1) < len(iter2):
                entail_dir = -1
            else:
                len(iter1) == len(iter2)
                entail_dir = 0
            
            return final_mapping, entail_dir
        else:
            # No valid permutations found, no mapping
            return None, None

    @staticmethod
    def isomorphic_conj_pair(iter1, iter2):
        """
        Recursive helper method for checking whether two nested lists of Literals
        with arbitrary depths are isomorphic
        """
        leaves1 = {l for l in iter1 if isinstance(l, Literal)}
        leaves2 = {l for l in iter2 if isinstance(l, Literal)}

        branches1 = [nc for nc in iter1 if not isinstance(nc, Literal)]
        branches2 = [nc for nc in iter2 if not isinstance(nc, Literal)]

        if Literal.entailing_mapping_btw(leaves1, leaves2)[1] != 0:
            # Not isomorphic if sets of leaf nodes are not isomorphic
            return False

        if len(branches1) != len(branches2):
            # Not isomorphic if sets of branch nodes have different lengths
            return False

        # Consider all possible bijections between branches1 vs. branches2;
        # if any bijection is isomorphic, return True
        for pm in permutations(range(len(branches1))):
            bijections = [(branches1[i], branches2[j]) for i, j in enumerate(pm)]

            if all(Literal.isomorphic_conj_pair(b1, b2) for b1, b2 in bijections):
                return True

        # If reached here, no isomorphic bijection found
        return False
