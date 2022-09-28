import re
import math
from itertools import product
from collections import defaultdict

import numpy as np

from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import logit, sigmoid, wrap_args


P_C = 0.01          # Catchall hypothesis probability
LOWER_THRES = 0.5   # Lower threshold for predicates that deserve closer look (see
                    # ../theoretical_reasoning/api.py)

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

    def export_reasoning_program(self, vis_scene):
        """
        Returns an ASP program that implements deductive & abductive reasonings by
        virtue of the listed entries
        """
        inference_prog = Program()

        # Add rules implementing deductive inference
        ded_out = self._add_deductive_inference_rules(inference_prog, vis_scene)
        grd_rule_probs, entries_by_head, intermediate_outputs = ded_out

        # Add rules implementing abductive inference
        self._add_abductive_inference_rules(
            inference_prog, vis_scene,
            grd_rule_probs, entries_by_head, intermediate_outputs
        )

        # Set of predicates that warrant consideration as possibility even with score
        # below the threshold because they're mentioned in KB
        preds_in_kb = set(self.entries_by_pred)
        preds_in_kb = {
            cat_type: {
                int(name.strip(f"{cat_type}_")) for name in preds_in_kb
                if name.startswith(cat_type)
            }
            for cat_type in ["cls", "att", "rel"]
        }

        return inference_prog, preds_in_kb

    def _add_deductive_inference_rules(self, inference_prog, vis_scene):
        """
        As self.export_reasoning_program() was getting too long, refactored code for
        deductive inference program synthesis from KB
        """
        # For storing rule head & body probabilities computed according to provided
        # vis_scene
        grd_rule_probs = []

        # For collecting entries by same heads, so that abductive inference rules can
        # be implemented for each collection
        entries_by_head = defaultdict(list)

        # For caching intermediate outputs assembled during the first (deductive) part
        # and reusing in the second (abductive) part
        intermediate_outputs = []

        # Helper method for fetching scores appropriate for literals from vis_scene
        def fetch_vis_score(pred, grd_args):
            cat_type, conc_ind = pred.split("_")
            conc_ind = int(conc_ind)

            # Fetch visual confidence score for category prediction with args
            obj = vis_scene[grd_args[0]]
            if cat_type == "cls" or cat_type == "att":
                if cat_type == "cls":
                    score = float(obj["pred_classes"][conc_ind])
                else:
                    score = float(obj["pred_attributes"][conc_ind])
                score = score if score > LOWER_THRES else 0.0
            else:
                assert cat_type == "rel"
                if "pred_relations" in obj and grd_args[1] in obj["pred_relations"]:
                    rels_per_obj = obj["pred_relations"][grd_args[1]]
                    score = float(rels_per_obj[conc_ind])
                    score = score if score > LOWER_THRES else 0.0
                else:
                    # No need to go further, return 0
                    score = 0.0

            return score

        # Process each entry
        for i, (rules, _) in enumerate(self.entries):
            # Storage of grounded head/body/rule probabilities
            h_grd_probs = {}; b_grd_probs = {}; r_grd_probs = {}

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

            inference_prog.add_hard_rule(
                Rule(head=h_sat_lit, body=h_sat_conds_pure+h_fn_assign)
            )
            inference_prog.add_hard_rule(
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
                    inference_prog.add_rule(Rule(
                        head=Literal(
                            f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[ft_lifted],
                            conds=rel_conds
                        ),
                        ub=1
                    ))
            add_assignment_choices(h_fn_terms, h_sat_conds)
            add_assignment_choices(b_fn_terms, b_sat_conds)

            # Rule violation flag
            r_unsat_lit = Literal(f"deduc_viol_{i}", wrap_args(*b_var_signature))
            inference_prog.add_hard_rule(Rule(
                head=r_unsat_lit, body=[h_sat_lit.flip(), b_sat_lit]
            ))

            # Helper method for computing prior probabilities of grounded head/body
            def compute_prior_prob(subs, sat_conds_pure, sat_conds_nonpure):
                # Value to return - update by multiplying values starting from 1.0
                prior = 1.0

                # "Pure" condition literals that don't include any function terms;
                # can be fetched directly from the provided vis_scene
                for lit in sat_conds_pure:
                    pred = lit.name
                    grd_args = [subs[a[0]] for a in lit.args]
                    score = fetch_vis_score(pred, grd_args)

                    if score == 0.0: return 0.0     # Short-circuit if score is zero
                    prior *= score

                # "Non-pure" condition literals are trickier; compute probabilities
                # that the skolem functions, which have existential readings, can assign
                # values with respect to the grounding of variables as specified by subs
                lit_unsat_probs = []
                for lit in sat_conds_nonpure:
                    pred = lit.name

                    # Flags for variables which stand for skolem function terms and thus
                    # cannot be grounded
                    still_var_args = [a[0] not in subs for a in lit.args]

                    # Probability that none of the full grounding of this literal will hold
                    l_unsat_prob = 1.0
                    for assig in product(vis_scene, repeat=sum(still_var_args)):
                        # Collect fully grounded arguments for the provided grounding
                        # substitution and this function value assignment
                        assig = list(assig); grd_args = []
                        for a, still_var in zip(lit.args, still_var_args):
                            if still_var: grd_args.append(assig.pop(-1))
                            else: grd_args.append(subs[a[0]])
                        
                        # Multiply unsat_prob by (1-score)
                        l_unsat_prob *= 1 - fetch_vis_score(pred, grd_args)
                    lit_unsat_probs.append(l_unsat_prob)
                
                # Multiply prior by prod{(1-P(L_i))}, which equals value of formula
                # obtained by the inclusion-exclusion principle
                prior *= math.prod([1-s for s in lit_unsat_probs])

                return prior

            # Collect probabilities of possible grounded rule heads & bodies,
            # depending on how they are instantiated and how functions map their
            # arguments to possible values
            possible_instantiations = product(vis_scene, repeat=len(b_var_signature))
            for inst in possible_instantiations:
                # Fetch prior probability value; compute if non-existent yet
                h_inst = tuple(inst[b_var_signature.index(a)] for a in h_var_signature)
                if h_inst not in h_grd_probs:
                    h_subs = {arg: ent for arg, ent in zip(h_var_signature, h_inst)}
                    h_grd_probs[h_inst] = compute_prior_prob(
                        h_subs, h_sat_conds_pure, h_sat_conds_nonpure
                    )

                b_inst = inst
                if b_inst not in b_grd_probs:
                    b_subs = {arg: ent for arg, ent in zip(b_var_signature, b_inst)}
                    b_grd_probs[b_inst] = compute_prior_prob(
                        b_subs, b_sat_conds_pure, b_sat_conds_nonpure
                    )

                r_grd_probs[inst] = (h_grd_probs[h_inst], b_grd_probs[b_inst])

            # Collect grounded proabilities by grounded rule instances
            grd_rule_probs.append(r_grd_probs)

            # Store intermediate outputs for later reuse
            intermediate_outputs.append((
                h_sat_lit, b_sat_lit, h_var_signature, b_var_signature
            ))

        # Manipulation of weight sums of models where rule is violated, i.e. body holds
        # but head does not. Manipulation weights are determined based on probabilities
        # of rule head literals and thus depend on grounding.
        for i, grd_probs in enumerate(grd_rule_probs):
            # Fetch rule probability; for rules with multiple provenance, probabilities
            # are aggregated by summing in logit-space and then sigmoid-ing back to
            # probability space
            provenances = self.entries[i][1]
            r_pr = sigmoid(sum(logit(w) for w, _ in provenances))

            b_var_signature = intermediate_outputs[i][3]

            for inst, (h_grd_pr, _) in grd_probs.items():
                # If prior Pr(Head) is smaller than rule probability, boost Pr(Head|Body)
                # up to the probability while retaining Pr(Body)
                if h_grd_pr < r_pr:
                    # Manipulate prior prob of grounded head for target conditional,
                    # while retaining distributions outside the conditional
                    grd_r_unsat_lit = Literal(f"deduc_viol_{i}", wrap_args(*inst))
                    grd_b_sat_lit = Literal(f"body_sat_{i}", wrap_args(*inst))

                    # Manipulating & balancing rule weights; effectively 'apply pressure'
                    # to model weights, so that the prior probability that head not being
                    # satisfied becomes (1-r_pr) instead of (1-h_grd_pr)
                    manipulator = logit(1-r_pr) - logit(1-h_grd_pr)
                    balancer = np.log(h_grd_pr) - np.log(r_pr)

                    # manipulator ::  :- not deduc_viol_i(*b_var_signature).
                    inference_prog.add_rule(
                        Rule(body=grd_r_unsat_lit.flip()), sigmoid(manipulator)
                    )
                    # balancer ::  :- not body_sat(*b_var_signature).
                    inference_prog.add_rule(
                        Rule(body=grd_b_sat_lit.flip()), sigmoid(balancer)
                    )

        return grd_rule_probs, entries_by_head, intermediate_outputs

    def _add_abductive_inference_rules(
        self, inference_prog, vis_scene, grd_rule_probs, entries_by_head, intermediate_outputs
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
                grd_probs = grd_rule_probs[ei]

                if ism is not None:
                    h_sat_lit = h_sat_lit.substitute(**ism)
                    b_sat_lit = b_sat_lit.substitute(**ism)

                    h_var_signature = [ism["terms"][v] for v in h_var_signature]
                    b_var_signature = [ism["terms"][v] for v in b_var_signature]

                standardized_outputs.append((
                    h_sat_lit, b_sat_lit, h_var_signature, b_var_signature, grd_probs
                ))

            coll_h_var_signature = standardized_outputs[0][2]

            # Index-neutral flag holding when any (and all) of the explanandum (head(s))
            # in the collection holds
            coll_h_sat_lit = Literal(
                f"coll_head_sat_{i}", wrap_args(*coll_h_var_signature)
            )

            for s_out in standardized_outputs:
                # coll_h_sat_lit holds when any (and all) of the heads hold
                inference_prog.add_hard_rule(Rule(head=coll_h_sat_lit, body=s_out[0]))

            # Flag holding when the explanandum (head) is not explained by any of
            # the explanantia (bodies), and thus evoke 'catchall' hypothesis
            coll_h_catchall_lit = Literal(
                f"abduc_catchall_{i}", wrap_args(*coll_h_var_signature)
            )

            # r_catchall_lit holds when coll_H_sat_lit holds but none of the
            # explanantia (bodies) hold
            unexpl_lits = [s_out[1].flip() for s_out in standardized_outputs]
            inference_prog.add_hard_rule(Rule(
                head=coll_h_catchall_lit, body=[coll_h_sat_lit]+unexpl_lits
            ))

            # Manipulation of weight sums of models where rule head is unexplained
            # and catchall hypothesis must be activated, i.e. head holds but none of
            # the explanantia (bodies for the head) hold. Manipulation weights are
            # determined based on probabilities of rule body literals and thus depend
            # on grounding.
            # (For now we will assume the body events are independent)
            h_possible_instantiations = product(vis_scene, repeat=len(coll_h_var_signature))
            for h_inst in h_possible_instantiations:
                # Collect probabilities that none of the bodies hold
                catchall_prob = 1.0
                for s_out in standardized_outputs:
                    h_var_signature = s_out[2]; b_var_signature = s_out[3]
                    grd_probs = s_out[4]

                    h_var_inds = [b_var_signature.index(v) for v in h_var_signature]

                    catchall_prob *= math.prod(
                        1-b_grd_prob for inst, (_, b_grd_prob) in grd_probs.items()
                        if tuple(inst[vi] for vi in h_var_inds) == h_inst
                    )

                    if catchall_prob == 0.0: break     # Short-circuit if prob is zero

                # If prior Pr(Catchall(<=>"None of the body holds")) is larger than the
                # catchall hypothesis probability, suppress Pr(Catchall|Head) down to the
                # probability while retaining Pr(Head)
                if catchall_prob > P_C:
                    # Manipulate prior prob of grounded body for target conditional,
                    # while retaining distributions outside the conditional
                    grd_coll_h_catchall_lit = Literal(
                        f"abduc_catchall_{i}", wrap_args(*h_inst)
                    )
                    grd_coll_h_sat_lit = Literal(
                        f"coll_head_sat_{i}", wrap_args(*h_inst)
                    )

                    # Manipulating & balancing rule weights
                    manipulator = logit(P_C) - logit(catchall_prob)
                    balancer = np.log(1-catchall_prob) - np.log(1-P_C)

                    # manipulator ::  :- not abduc_catchall_i(*b_var_signature).
                    inference_prog.add_rule(
                        Rule(body=grd_coll_h_catchall_lit.flip()), sigmoid(manipulator)
                    )
                    # balancer ::  :- not coll_head_sat(*h_var_signature).
                    inference_prog.add_rule(
                        Rule(body=grd_coll_h_sat_lit.flip()), sigmoid(balancer)
                    )
