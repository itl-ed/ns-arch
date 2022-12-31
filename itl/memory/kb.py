import re
from collections import defaultdict

from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import wrap_args, flatten_head_body, unify_mappings


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

        # Neutralizing variable & function names by stripping off turn/clause
        # indices, etc.
        rename_var = {
            a for a, _ in _flatten(_extract_terms(head+body))
            if isinstance(a, str)
        }
        rename_var = {
            (vn, True): (re.match("(.+?)(t\d+c\d+)?$", vn).group(1), True)
            for vn in rename_var
        }
        rename_fn = {
            a[0] for a, _ in _flatten(_extract_terms(head+body))
            if isinstance(a, tuple)
        }
        rename_fn = {
            fn: re.match("(.+?)(_.+)?$", fn).group(1)+f"_{i}"
            for i, fn in enumerate(rename_fn)
        }
        head = tuple(_substitute(h, rename_var, rename_fn) for h in head)
        body = tuple(_substitute(b, rename_var, rename_fn) for b in body)

        preds_head = set(_flatten(_extract_preds(head)))
        preds_body = set(_flatten(_extract_preds(body)))

        # Check if the input knowledge can be logically entailed by some existing
        # KB entry (or is already contained in the KB). For now, just check
        # simple one-step entailments (by A,B |- A).

        # Initial filtering of irrelevant entries without any overlapping for
        # both head and body
        entries_with_overlap = set.union(*[
            self.entries_by_pred.get(pred, set()) for pred in preds_head
        ]) & set.union(*[
            self.entries_by_pred.get(pred, set()) for pred in preds_body
        ])

        entries_entailed = set()       # KB entries entailed by input
        entries_entailing = set()      # KB entries that entail input
        for ent_id in entries_with_overlap:
            (ent_head, ent_body), ent_weight, _ = self.entries[ent_id]

            # Find (partial) term mapping between the KB entry and input with
            # which they can unify
            mapping_b, ent_dir_b = Literal.entailing_mapping_btw(
                body, ent_body
            )
            if mapping_b is not None:
                mapping_h, ent_dir_h = Literal.entailing_mapping_btw(
                    head, ent_head, mapping_b
                )
                if mapping_h is not None and {ent_dir_h, ent_dir_b} != {1, -1}:
                    # Entailment relation detected
                    if ent_dir_h >= 0 and ent_dir_b <= 0 and weight >= ent_weight:
                        entries_entailed.add(ent_id)
                    if ent_dir_h <= 0 and ent_dir_b >= 0 and weight <= ent_weight:
                        entries_entailing.add(ent_id)

        if len(entries_entailing) == len(entries_entailed) == 0:
            # Add the input as a whole new entry along with the weight & source
            # and index it by occurring predicates
            self.entries.append(((head, body), weight, [(source, weight)]))
            for pred in preds_head | preds_body:
                self.entries_by_pred[pred].add(len(self.entries)-1)
            
            kb_updated = True
        else:
            # Due to the invariant condition that there's no two KB entries such
            # that one is strictly stronger than the other, the input shouldn't be
            # entailed by some entries and entail others at the same time -- except
            # for the case of exact match.
            if len(entries_entailing) > 0 and len(entries_entailed) > 0:
                assert entries_entailing == entries_entailed
                # Equivalent entry exists; just add to provenance list
                for ent_id in entries_entailing:
                    self.entries[ent_id][2].append((source, weight))

                kb_updated = False

            else:
                if len(entries_entailed) > 0:
                    # Stronger input entails some KB entries and render them
                    # 'obsolete'; the entailed entries may be removed and merged
                    # into the newly added entry
                    self.remove_by_ids(entries_entailed)

                    # Add the stronger input as new entry
                    self.entries.append(
                        ((head, body), weight, [(source, weight)])
                    )
                    for pred in preds_head | preds_body:
                        self.entries_by_pred[pred].add(len(self.entries)-1)

                    kb_updated = True
                else:
                    assert len(entries_entailing) > 0
                    # Entry entailing the given input exists; just add to provenance list
                    for ent_id in entries_entailing:
                        self.entries[ent_id][2].append((source, weight))

                    kb_updated = False

        return kb_updated

    def remove_by_ids(self, ent_ids):
        """
        Update KB by removing entries designated by the list of entry ids provided
        """
        # First find the mapping from previous set of indices to new set of indices,
        # as indices will change as entries shift their positions to fill in blank
        # positions
        ind_map = {}; ni = 0
        for ent_id in range(len(self.entries)):
            if ent_id in ent_ids:
                ni += 1
            else:
                ind_map[ent_id] = ent_id - ni

        # Cull the specified entries and update entry indexing by predicate
        # (self.entries_by_pred) according to the mapping found above
        self.entries = [
            entry for ent_id, entry in enumerate(self.entries)
            if ent_id in ind_map
        ]
        self.entries_by_pred = defaultdict(set, {
            pred: {ind_map[ei] for ei in ent_ids if ei in ind_map}
            for pred, ent_ids in self.entries_by_pred.items()
        })

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
        for i, (rule, weight, _) in enumerate(self.entries):
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
                    ism, ent_dir = Literal.entailing_mapping_btw(head, h_lits)
                    if ent_dir == 0:
                        entries_by_head[h_lits].append((i, ism))
                        break
                else:
                    entries_by_head[frozenset(head)].append((i, None))

            # Choice rule for function value assignments
            def add_assignment_choices(fn_terms, sat_conds):
                for ft in fn_terms:
                    # Function arguments and function term lifted
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
            # against deductive rule violation
            inference_prog.add_rule(Rule(body=r_unsat_lit), weight)

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
_extract_terms = lambda cnjt: cnjt.args if isinstance(cnjt, Literal) \
    else [_extract_terms(nc) for nc in cnjt]
_extract_preds = lambda cnjt: cnjt.name if isinstance(cnjt, Literal) \
    else [_extract_preds(nc) for nc in cnjt]
_substitute = lambda cnjt, ts, fs: cnjt.substitute(terms=ts, functions=fs) \
    if isinstance(cnjt, Literal) else [_substitute(nc, ts, fs) for nc in cnjt]
def _flatten(ls):
    for x in ls:
        if isinstance(x, list):
            yield from _flatten(x)
        else:
            yield x
