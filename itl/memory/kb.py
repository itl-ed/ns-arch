import re
from collections import defaultdict

from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import logit, sigmoid, wrap_args


P_C = 0.01          # Catchall hypothesis probability
LOWER_THRES = 0.5   # Lower threshold for predicates that deserve closer look (see
                    # ../symbolic_reasoning/api.py)

class KnowledgeBase:
    """
    Knowledge base containing pieces of knowledge in the form of LP^MLN (weighted
    ASP) rules
    """
    def __init__(self):
        # Knowledge base entries stored as collections of rules; each collection
        # corresponds to a batch (i.e. conjunction) of generic rules extracted from
        # the same provenance (utterances from teacher, etc.), associated with a
        # shared probability weight value between 0 ~ 1. Each collection stands for
        # a conjunction of heads of rules, which should share the same rule body.
        self.entries = []

        # Indexing entries by contained predicates
        self.entries_by_pred = defaultdict(set)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"KnowledgeBase(len={len(self)})"

    def add(self, rules, weight, source):
        """ Returns whether KB is expanded or not """
        # More than one rules should represent conjunction of rule heads for the same
        # rule body
        assert all(r.body == rules[0].body for r in rules)

        # Neutralizing variable & function names by stripping off utterance indices, etc.
        rename_var = set.union(*[
            {t[0] for t in r.terms() if type(t[0])==str} for r in rules
        ])
        rename_var = {
            (vn, True): (re.match("(.+)u.*$", vn).group(1), True) for vn in rename_var
        }
        rename_fn = set.union(*[
            {t[0][0] for t in r.terms() if type(t[0])==tuple} for r in rules
        ])
        rename_fn = {
            fn: re.match("(.+)_.*$", fn).group(1)+str(i) for i, fn in enumerate(rename_fn)
        }
        rules = { r.substitute(terms=rename_var, functions=rename_fn) for r in rules }

        occurring_preds = set.union(*[{lit.name for lit in r.literals()} for r in rules])

        # Check if the input knowledge is already contained in the KB
        matched_entry_id = None
        entries_with_overlap = set.union(*[
            self.entries_by_pred[pred] for pred in occurring_preds
        ])          # Initial filtering out entries without any overlapping

        for ent_id in entries_with_overlap:
            ent_rules, _ = self.entries[ent_id]

            # Don't even bother with different set sizes
            if len(rules) != len(ent_rules): continue

            no_match = False
            rules_proxy = [r for r in rules]; ent_rules_proxy = [er for er in ent_rules]
            while len(rules_proxy) > 0:
                r = rules_proxy.pop()
                for er in ent_rules_proxy:
                    if r.is_isomorphic_to(er):
                        ent_rules_proxy.remove(er)
                        break
                else:
                    no_match = True

                # Break if there's a rule that's not isomorphic to any
                if no_match: break

            if len(rules_proxy) == len(ent_rules_proxy) == 0:
                # Match found; record ent_id and break
                matched_entry_id = ent_id
                break

        is_new_entry = matched_entry_id is None
        if is_new_entry:
            # If not contained, add the rules as a new entry along with the weight-source
            # pair and index it by predicate
            self.entries.append((rules, {(weight, source)}))
            for pred in occurring_preds:
                self.entries_by_pred[pred].add(len(self.entries)-1)
        else:
            # If already contained, add to existing set of weight-source pairs for the rules
            # (only if from new source)
            _, metadata = self.entries[matched_entry_id]
            existing_sources = {src for _, src in metadata}

            if source not in existing_sources:
                metadata.add((weight, source))

        return is_new_entry

    def export_reasoning_program(self):
        """
        Returns an ASP program that implements deductive & abductive reasonings by
        virtue of the listed entries
        """
        inference_prog = Program()

        # Add rules implementing deductive inference
        entries_by_head, intermediate_outputs = \
            self._add_deductive_inference_rules(inference_prog)

        # Add rules implementing abductive inference
        self._add_abductive_inference_rules(
            inference_prog, entries_by_head, intermediate_outputs
        )

        # Set of predicates that warrant consideration as possibility even with score
        # below the threshold because they're mentioned in KB
        preds_in_kb = set(self.entries_by_pred)
        preds_in_kb = {
            conc_type: {
                int(name.strip(f"{conc_type}_")) for name in preds_in_kb
                if name.startswith(conc_type)
            }
            for conc_type in ["cls", "att", "rel"]
        }

        return inference_prog, preds_in_kb

    def _add_deductive_inference_rules(self, inference_prog):
        """
        As self.export_reasoning_program() was getting too long, refactored code for
        deductive inference program synthesis from KB
        """
        # For collecting entries by same heads, so that abductive inference rules can
        # be implemented for each collection
        entries_by_head = defaultdict(list)

        # For caching intermediate outputs assembled during the first (deductive) part
        # and reusing in the second (abductive) part
        intermediate_outputs = []

        # Process each entry
        for i, (rules, _) in enumerate(self.entries):
            all_fn_args = set(); all_var_names = set()
            for r in rules:
                # All function term args used in this rule
                all_fn_args |= {
                    a for a in set.union(*[set(l.args) for l in r.literals()])
                    if type(a[0])==tuple
                }

                # Keep track of variable names used to avoid accidentally using
                # overlapping names for 'lifting' variables (see below)
                all_var_names |= {
                    t_val
                    for t_val, t_is_var in set.union(*[l.nonfn_terms() for l in r.literals()])
                    if t_is_var
                }

            # Attach unique identifier suffixes to function names, so that functions
            # from different KB entries can be distinguished; names are shared across
            # within entry
            all_fn_names = {a[0][0] for a in all_fn_args}
            fn_name_map = { fn: f"{fn}_{i}" for fn in all_fn_names }

            # Map for lifting function term to new variable arg term
            fn_lifting_map = {
                ((fn_name_map[fa[0][0]], fa[0][1]), fa[1]):
                    (f"X{i+len(all_var_names)}", True)
                for i, fa in enumerate(all_fn_args)
            }

            rules_fn_subs = [r.substitute(functions=fn_name_map) for r in rules]
            rules_lifted = [r.substitute(terms=fn_lifting_map) for r in rules_fn_subs]

            # List of unique non-function variable arguments in 1) rule head and 2) rule body
            # (effectively whole rule as well) in the order of occurrence
            h_var_signature = []; b_var_signature = []
            for r in rules_fn_subs:
                for hl in r.head:
                    for v_val, _ in hl.nonfn_terms():
                        if v_val not in h_var_signature: h_var_signature.append(v_val)
                for bl in r.body:
                    for v_val, _ in bl.nonfn_terms():
                        if v_val not in b_var_signature: b_var_signature.append(v_val)

            # Rule head/body satisfaction flags literals
            h_sat_lit = Literal(f"head_sat_{i}", wrap_args(*h_var_signature))
            b_sat_lit = Literal(f"body_sat_{i}", wrap_args(*b_var_signature))

            # Flag literal is derived when head/body is satisfied; in the meantime, lift
            # occurrences of function terms and add appropriate function value assignment
            # literals
            h_sat_conds = list(set.union(*[set(r.head) for r in rules_lifted]))
            h_sat_conds_pure = [         # Conditions having only 'pure' non-function args
                lit for lit in h_sat_conds
                if all(a[0] in h_var_signature for a in lit.args)
            ]
            h_sat_conds_nonpure = [      # Conditions having some function args
                lit for lit in h_sat_conds
                if any(a[0] not in h_var_signature for a in lit.args)
            ]
            h_fn_terms = set.union(*[
                set.union(*[{a for a in hl.args if type(a[0])==tuple} for hl in r.head])
                for r in rules_fn_subs
            ])
            h_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in h_fn_terms
            ]

            b_sat_conds = list(set.union(*[set(r.body) for r in rules_lifted]))
            b_sat_conds_pure = [
                lit for lit in b_sat_conds
                if all(a[0] in b_var_signature for a in lit.args)
            ]
            b_sat_conds_nonpure = [
                lit for lit in b_sat_conds
                if any(a[0] not in b_var_signature for a in lit.args)
            ]
            b_fn_terms = set.union(*[
                set.union(*[{a for a in bl.args if type(a[0])==tuple} for bl in r.body])
                for r in rules_fn_subs
            ])
            b_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in b_fn_terms
            ]

            inference_prog.add_absolute_rule(
                Rule(head=h_sat_lit, body=h_sat_conds_pure+h_fn_assign)
            )
            inference_prog.add_absolute_rule(
                Rule(head=b_sat_lit, body=b_sat_conds_pure+b_fn_assign)
            )

            # Indexing & storing the entry by head
            hd_content = set.union(*[set(hl for hl in r.head) for r in rules])
            for h_lits in entries_by_head:
                ism = Literal.isomorphism_btw(hd_content, h_lits, None)
                if ism is not None:
                    entries_by_head[h_lits].append((i, ism))
                    break
            else:
                entries_by_head[frozenset(hd_content)].append((i, None))

            # Choice rule for function value assignments
            def add_assignment_choices(fn_terms, sat_conds):
                for ft in fn_terms:
                    # Function arguments and function term lifted
                    fn_args = wrap_args(*ft[0][1])
                    ft_lifted = fn_lifting_map[ft]

                    # Filter relevant conditions for filtering options worth considering
                    rel_conds = [
                        cl for cl in sat_conds
                        if ft_lifted in cl.args or any(fa in cl.args for fa in fn_args)
                    ]
                    inference_prog.add_absolute_rule(
                        Rule(
                            head=Literal(
                                f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[ft_lifted]
                            ),
                            body=rel_conds
                        )
                    )
            add_assignment_choices(h_fn_terms, h_sat_conds)
            add_assignment_choices(b_fn_terms, b_sat_conds)

            # Rule violation flag
            r_unsat_lit = Literal(f"deduc_viol_{i}", wrap_args(*b_var_signature))
            inference_prog.add_absolute_rule(Rule(
                head=r_unsat_lit, body=[h_sat_lit.flip(), b_sat_lit]
            ))
            
            # Add appropriately weighted rule for applying 'probabilistic pressure'
            # against to deductive rule violation
            provenances = self.entries[i][1]
            r_pr = sigmoid(sum(logit(w) for w, _ in provenances))
            inference_prog.add_rule(Rule(body=r_unsat_lit), r_pr)

            # Store intermediate outputs for later reuse
            intermediate_outputs.append((
                h_sat_lit, b_sat_lit, h_var_signature, b_var_signature
            ))

        return entries_by_head, intermediate_outputs

    def _add_abductive_inference_rules(
        self, inference_prog, entries_by_head, intermediate_outputs
    ):
        """
        As self.export_reasoning_program() was getting too long, refactored code for
        abductive inference program synthesis from KB
        """
        for i, entry_collection in enumerate(entries_by_head.values()):
            # (If there are more than one entries in collection) Standardize names
            # to comply with the first entry in collection, using the discovered
            # isomorphic mappings (which should not be None)
            standardized_outputs = []
            for ei, ism in entry_collection:
                h_sat_lit, b_sat_lit, h_var_signature, b_var_signature \
                    = intermediate_outputs[ei]

                if ism is not None:
                    h_sat_lit = h_sat_lit.substitute(**ism)
                    b_sat_lit = b_sat_lit.substitute(**ism)

                    h_var_signature = [ism["terms"][v] for v in h_var_signature]
                    b_var_signature = [ism["terms"][v] for v in b_var_signature]

                standardized_outputs.append((
                    h_sat_lit, b_sat_lit, h_var_signature, b_var_signature
                ))

            coll_h_var_signature = standardized_outputs[0][2]

            # Index-neutral flag holding when any (and all) of the explanandum (head(s))
            # in the collection holds
            coll_h_sat_lit = Literal(
                f"coll_head_sat_{i}", wrap_args(*coll_h_var_signature)
            )

            for s_out in standardized_outputs:
                # coll_h_sat_lit holds when any (and all) of the heads hold
                inference_prog.add_absolute_rule(
                    Rule(head=coll_h_sat_lit, body=s_out[0])
                )

            # Flag holding when the explanandum (head) is not explained by any of
            # the explanantia (bodies), and thus evoke 'catchall' hypothesis
            coll_h_catchall_lit = Literal(
                f"abduc_catchall_{i}", wrap_args(*coll_h_var_signature)
            )

            # r_catchall_lit holds when coll_h_sat_lit holds but none of the
            # explanantia (bodies) hold
            unexpl_lits = [s_out[1].flip() for s_out in standardized_outputs]
            inference_prog.add_absolute_rule(Rule(
                head=coll_h_catchall_lit, body=[coll_h_sat_lit]+unexpl_lits
            ))

            # Add appropriately weighted rule for applying 'probabilistic pressure'
            # against resorting to catchall hypothesis due to absence of abductive
            # explanation of head
            inference_prog.add_rule(Rule(body=coll_h_catchall_lit), 1-P_C)
