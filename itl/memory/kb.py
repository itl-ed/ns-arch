import re
from collections import defaultdict

from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import logit, sigmoid, wrap_args, flatten_head_body


P_C = 0.01          # Catchall hypothesis probability

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

    def __contains__(self, item):
        """ Test if an entry isomorphic to item exists """
        head, body = item

        for (ent_head, ent_body), _ in self.entries:
            # Don't even bother with different set sizes
            if len(head) != len(ent_head): continue
            if len(body) != len(ent_body): continue

            head_isomorphic = Literal.isomorphic_conj_pair(head, ent_head)
            body_isomorphic = Literal.isomorphic_conj_pair(body, ent_body)

            if head_isomorphic and body_isomorphic:
                return True

        return False

    def add(self, rule, weight, source):
        """ Returns whether KB is expanded or not """
        head, body = rule

        # Neutralizing variable & function names by stripping off turn/clause indices, etc.
        rename_var = {
            a for a, _ in _flatten(_extract_terms(head+body)) if isinstance(a, str)
        }
        rename_var = {
            (vn, True): (re.match("(.+?)(t\d+c\d+)?$", vn).group(1), True) for vn in rename_var
        }
        rename_fn = {
            a[0] for a, _ in _flatten(_extract_terms(head+body)) if isinstance(a, tuple)
        }
        rename_fn = {
            fn: re.match("(.+?)(_.+)?$", fn).group(1)+f"_{i}" for i, fn in enumerate(rename_fn)
        }
        head = tuple(_substitute(h, rename_var, rename_fn) for h in head)
        body = tuple(_substitute(b, rename_var, rename_fn) for b in body)

        occurring_preds = set(_flatten(_extract_preds(head+body)))

        # Check if the input knowledge is already contained in the KB
        matched_entry_id = None
        entries_with_overlap = set.intersection(*[
            self.entries_by_pred[pred] for pred in occurring_preds
        ])          # Initial filtering out entries without any overlapping

        for ent_id in entries_with_overlap:
            (ent_head, ent_body), _ = self.entries[ent_id]

            # Don't even bother with different set sizes
            if len(head) != len(ent_head): continue
            if len(body) != len(ent_body): continue

            head_isomorphic = Literal.isomorphic_conj_pair(head, ent_head)
            body_isomorphic = Literal.isomorphic_conj_pair(body, ent_body)

            if head_isomorphic and body_isomorphic:
                # Match found; record ent_id and break
                matched_entry_id = ent_id
                break

        is_new_entry = matched_entry_id is None
        if is_new_entry:
            # If not contained, add the rules as a new entry along with the weight-source
            # pair and index it by predicate
            self.entries.append(((head, body), {(weight, source)}))
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
        for i, (rule, _) in enumerate(self.entries):
            head, body = flatten_head_body(*rule)

            # Keep track of variable names used to avoid accidentally using
            # overlapping names for 'lifting' variables (see below)
            all_var_names = {
                a for a, _ in _flatten(_extract_terms(head+body))
                if isinstance(a, str)
            }

            # All function term args used in this rule
            all_fn_args = {
                a for a in _flatten(_extract_terms(head+body))
                if isinstance(a[0], tuple)
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

            rule_fn_subs = {
                "head": [h.substitute(functions=fn_name_map) for h in head],
                "body": [b.substitute(functions=fn_name_map) for b in body]
            }
            rule_lifted = {
                "head": [h.substitute(terms=fn_lifting_map) for h in rule_fn_subs["head"]],
                "body": [b.substitute(terms=fn_lifting_map) for b in rule_fn_subs["body"]]
            }

            # List of unique non-function variable arguments in 1) rule head and 2) rule body
            # (effectively whole rule) in the order of occurrence
            h_var_signature = []; b_var_signature = []
            for hl in rule_fn_subs["head"]:
                for v_val, _ in hl.nonfn_terms():
                    if v_val not in h_var_signature: h_var_signature.append(v_val)
            for bl in rule_fn_subs["body"]:
                for v_val, _ in bl.nonfn_terms():
                    if v_val not in b_var_signature: b_var_signature.append(v_val)

            # Rule head/body satisfaction flags literals
            h_sat_lit = Literal(f"head_sat_{i}", wrap_args(*h_var_signature))
            b_sat_lit = Literal(f"body_sat_{i}", wrap_args(*b_var_signature))

            # Flag literal is derived when head/body is satisfied; in the meantime, lift
            # occurrences of function terms and add appropriate function value assignment
            # literals
            h_sat_conds_pure = [        # Conditions having only 'pure' non-function args
                lit for lit in rule_lifted["head"]
                if all(a[0] in h_var_signature for a in lit.args)
            ]
            h_fn_terms = set.union(*[
                {a for a in l.args if type(a[0])==tuple} for l in rule_fn_subs["head"]
            ]) if len(rule_fn_subs["head"]) > 0 else set()
            h_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in h_fn_terms
            ]

            b_sat_conds_pure = [
                lit for lit in rule_lifted["body"]
                if all(a[0] in b_var_signature for a in lit.args)
            ]
            b_fn_terms = set.union(*[
                {a for a in l.args if type(a[0])==tuple} for l in rule_fn_subs["body"]
            ])
            b_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in b_fn_terms
            ]

            if len(h_sat_conds_pure+h_fn_assign) > 0:
                # Skip headless rules
                inference_prog.add_absolute_rule(
                    Rule(head=h_sat_lit, body=h_sat_conds_pure+h_fn_assign)
                )
            inference_prog.add_absolute_rule(
                Rule(head=b_sat_lit, body=b_sat_conds_pure+b_fn_assign)
            )

            # Indexing & storing the entry by head for later abductive rule
            # translation (thus, no need to consider headless constraints)
            if len(head) > 0:
                for h_lits in entries_by_head:
                    ism = Literal.isomorphism_btw(head, h_lits)
                    if ism is not None:
                        entries_by_head[h_lits].append((i, ism))
                        break
                else:
                    entries_by_head[frozenset(head)].append((i, None))

            # Choice rule for function value assignments
            def add_assignment_choices(fn_terms, sat_conds):
                for ft in fn_terms:
                    # Function arguments and function term lifted
                    fn_args = wrap_args(*ft[0][1])
                    ft_lifted = fn_lifting_map[ft]

                    # Filter relevant conditions for filtering options worth considering
                    rel_conds = [cl for cl in sat_conds if ft_lifted in cl.args]
                    inference_prog.add_absolute_rule(
                        Rule(
                            head=Literal(
                                f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[ft_lifted]
                            ),
                            body=rel_conds
                        )
                    )
            add_assignment_choices(h_fn_terms, rule_lifted["head"])
            add_assignment_choices(b_fn_terms, rule_lifted["body"])

            # Rule violation flag
            r_unsat_lit = Literal(f"deduc_viol_{i}", wrap_args(*b_var_signature))
            inference_prog.add_absolute_rule(Rule(
                head=r_unsat_lit,
                body=[b_sat_lit] + ([h_sat_lit.flip()] if len(head) > 0 else [])
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


# Recursive helper methods for fetching predicate terms and names, substituting
# variables and function names while preserving structure, and flattening nested
# lists with arbitrary depths into a single list
_extract_terms = lambda cnj: cnj.args if isinstance(cnj, Literal) \
    else [_extract_terms(nc) for nc in cnj]
_extract_preds = lambda cnj: cnj.name if isinstance(cnj, Literal) \
    else [_extract_preds(nc) for nc in cnj]
_substitute = lambda cnj, ts, fs: cnj.substitute(terms=ts, functions=fs) \
    if isinstance(cnj, Literal) else [_substitute(nc, ts, fs) for nc in cnj]
def _flatten(ls):
    for x in ls:
        if isinstance(x, list):
            yield from _flatten(x)
        else:
            yield x
