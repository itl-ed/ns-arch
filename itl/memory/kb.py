import re
from itertools import product
from collections import defaultdict

import numpy as np

from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import logit, sigmoid, wrap_args


P_C = 0.01          # Catchall hypothesis probability
DEF_P_PR = 0.05     # Default prior probability for any predicates holding, prior to
                    # observation of any visual evidence

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

    def export_reasoning_program(self, vis_scene, category_thresh=0.75):
        """
        Returns an ASP program that implements deductive & abductive reasonings by
        virtue of the listed entries
        """
        inference_prog = Program()

        # For storing rule head & body probabilities computed according to provided
        # vis_scene
        head_probs = defaultdict(lambda: 1.0)
        body_probs = defaultdict(lambda: 1.0)

        # Add rules implementing deductive inference
        ded_out = self._add_deductive_inference_rules(
            inference_prog, vis_scene, category_thresh, head_probs, body_probs
        )
        entries_by_head, intermediate_outputs = ded_out

        # Add rules implementing abductive inference
        self._add_abductive_inference_rules(
            inference_prog, head_probs, body_probs,
            entries_by_head, intermediate_outputs
        )

        return inference_prog

    def _add_deductive_inference_rules(
        self, inference_prog, vis_scene, category_thresh, head_probs, body_probs
    ):
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

        for i, (rules, provenances) in enumerate(self.entries):
            # For rules with multiple provenance, probabilities are aggregated by
            # summing in logit-space and then sigmoid-ing back to probability space
            r_pr = sigmoid(sum(logit(w) for w, _ in provenances))

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

            # List of unique non-term arg vars in 1) rule head or 2) rule body (whole
            # rule, in effect) -- in the order of occurrence
            h_var_signature = []; b_var_signature = []
            for r in rules:
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
            h_sat_conds = list(set.union(*[
                set(hl.substitute(terms=fn_lifting_map) for hl in r.head)
                for r in rules_fn_subs
            ]))
            h_fn_terms = set.union(*[
                set.union(*[{a for a in hl.args if type(a[0])==tuple} for hl in r.head])
                for r in rules_fn_subs
            ])
            h_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in h_fn_terms
            ]

            b_sat_conds = list(set.union(*[
                set(bl.substitute(terms=fn_lifting_map) for bl in r.body)
                for r in rules_fn_subs
            ]))
            b_fn_terms = set.union(*[
                set.union(*[{a for a in bl.args if type(a[0])==tuple} for bl in r.body])
                for r in rules_fn_subs
            ])
            b_fn_assign = [
                Literal(f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[fn_lifting_map[ft]])
                for ft in b_fn_terms
            ]

            inference_prog.add_hard_rule(
                Rule(head=h_sat_lit, body=h_sat_conds+h_fn_assign)
            )
            inference_prog.add_hard_rule(
                Rule(head=b_sat_lit, body=b_sat_conds+b_fn_assign)
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

            # Choice rule for function value assignment; options that are ever worth
            # considering should satisfy both head and body
            for ft in (h_fn_terms | b_fn_terms):
                # Function arguments and function term lifted
                fn_args = wrap_args(*ft[0][1])
                ft_lifted = fn_lifting_map[ft]

                # Filter relevant conditions for filtering options worth considering
                rel_conds = [
                    cl for cl in h_sat_conds+b_sat_conds
                    if ft_lifted in cl.args or any(fa in cl.args for fa in fn_args)
                ]
                inference_prog.add_rule(Rule(
                    head=Literal(
                        f"assign_{ft[0][0]}", wrap_args(*ft[0][1])+[ft_lifted],
                        conds=rel_conds
                    ),
                    ub=1
                ))

            # Rule violation flag
            r_unsat_lit = Literal(f"deduc_viol_{i}", wrap_args(*b_var_signature))
            inference_prog.add_hard_rule(Rule(
                head=r_unsat_lit, body=[h_sat_lit.flip(), b_sat_lit]
            ))

            # Collect probabilities of possible grounded rule heads & bodies,
            # depending on how they are instantiated and how functions map their
            # arguments to possible values
            h_possible_instantiations = product(
                vis_scene, repeat=len(h_var_signature)+len(fn_lifting_map)
            )
            for h_inst in h_possible_instantiations:
                h_inst_var = h_inst[:len(h_var_signature)]
                h_inst_fn_assig = h_inst[len(h_var_signature):]

                # For each possible instantiations of rule head atoms...
                var_subs = {
                    arg: ent for arg, ent in zip(h_var_signature, h_inst_var)
                }
                fn_assigs = {
                    assig[0]: ent for assig, ent in zip(fn_lifting_map, h_inst_fn_assig)
                }

                for hl in h_sat_conds:
                    cat_type, conc_ind = hl.name.split("_")
                    conc_ind = int(conc_ind)

                    # Fetch visual confidence score for category prediction with args
                    obj = vis_scene[inst_subs[hl.args[0][0]]]
                    if cat_type == "cls" or cat_type == "att":
                        if cat_type == "cls":
                            prior_score = obj["pred_classes"][conc_ind]
                        else:
                            prior_score = obj["pred_attributes"][conc_ind]
                    else:
                        assert cat_type == "rel"
                        rels_per_obj = obj["pred_relations"][inst_subs[hl.args[1][0]]]
                        prior_score = rels_per_obj[conc_ind]

                    # Multiply to update total grounded head probability
                    prior_score = float(prior_score)
                    prior_score = prior_score \
                        if prior_score > category_thresh else DEF_P_PR
                    head_probs[(i, h_inst)] *= prior_score

            # Penalization of rule violation where body holds but head does not --
            # partially grounded according to body probability score values. Penalty
            # weights are computed based on probabilities or rule head atoms. 
            for h_inst in product(vis_scene, repeat=len(h_var_signature)):
                # For each possible instantiations of rule head atoms...
                inst_subs = {
                    arg: ent for arg, ent in zip(h_var_signature, h_inst)
                }

                grd_b_var_signature = [inst_subs[arg] for arg in b_var_signature]
                grd_r_unsat_lit = Literal(
                    f"deduc_viol_{i}", wrap_args(*grd_b_var_signature)
                )

                for hl in h_sat_conds:
                    cat_type, conc_ind = hl.name.split("_")
                    conc_ind = int(conc_ind)

                    # Fetch visual confidence score for category prediction with args
                    obj = vis_scene[inst_subs[hl.args[0][0]]]
                    if cat_type == "cls" or cat_type == "att":
                        if cat_type == "cls":
                            prior_score = obj["pred_classes"][conc_ind]
                        else:
                            prior_score = obj["pred_attributes"][conc_ind]
                    else:
                        assert cat_type == "rel"
                        rels_per_obj = obj["pred_relations"][inst_subs[hl.args[1][0]]]
                        prior_score = rels_per_obj[conc_ind]

                    # Multiply to update total grounded head probability
                    prior_score = float(prior_score)
                    prior_score = prior_score \
                        if prior_score > category_thresh else DEF_P_PR
                    head_probs[(i, h_inst)] *= prior_score
                
                if head_probs[(i, h_inst)] != r_pr:
                    # Manipulate prior prob of grounded head for target conditional,
                    # while retaining distributions outside the conditional

                    # Manipulating & balancing rule weights; effectively 'apply pressure'
                    # to model weights, so that the prior probability that head not being
                    # satisfied becomes (1-r_pr) instead of (1-head_prior)
                    manipulator = logit(1-r_pr) - logit(1-head_probs[(i, h_inst)])
                    balancer = np.log(head_probs[(i, h_inst)]) - np.log(r_pr)

                    # manipulator ::  :- not deduc_viol_i_j(*h_var_signature).
                    inference_prog.add_rule(
                        Rule(body=grd_r_unsat_lit.flip()), sigmoid(manipulator)
                    )
                    # balancer ::  :- not body_sat(*h_var_signature).
                    inference_prog.add_rule(
                        Rule(body=b_sat_lit.flip()), sigmoid(balancer)
                    )

            # Also prepare prior probabilities of rule body in advance, in preparation
            # of adding the abductive inference rules
            for b_inst in product(vis_scene, repeat=len(b_var_signature)):
                # For each possible instantiations of rule...
                inst_subs = {
                    arg: ent for arg, ent in zip(b_var_signature, b_inst)
                }

                for bl in b_sat_conds:
                    cat_type, conc_ind = bl.name.split("_")
                    conc_ind = int(conc_ind)

                    # Fetch visual confidence score for category prediction with args
                    obj = vis_scene[inst_subs[bl.args[0][0]]]
                    if cat_type == "cls" or cat_type == "att":
                        if cat_type == "cls":
                            prior_score = obj["pred_classes"][conc_ind]
                        else:
                            prior_score = obj["pred_attributes"][conc_ind]
                    else:
                        assert cat_type == "rel"
                        rels_per_obj = obj["pred_relations"][inst_subs[bl.args[1][0]]]
                        prior_score = rels_per_obj[conc_ind]

                    # Multiply to update total grounded body probability
                    prior_score = float(prior_score)
                    prior_score = prior_score \
                        if prior_score > category_thresh else DEF_P_PR
                    body_probs[(i, b_inst)] *= prior_score

            # Finally store useful outputs for reuse later
            intermediate_outputs.append((
                h_var_signature, b_var_signature, h_sat_lit, b_sat_lit
            ))

        return entries_by_head, intermediate_outputs

    def _add_abductive_inference_rules(
        self, inference_prog, head_probs, body_probs,
        entries_by_head, intermediate_outputs
    ):
        """
        As self.export_reasoning_program() was getting too long, refactored code for
        abductive inference program synthesis from KB
        """
        for i, (h_lits, entry_collection) in enumerate(entries_by_head.items()):
            # (If there are more than one entries in collection) Standardize names
            # to comply with the first entry in collection, using the discovered
            # isomorphic mappings (which should not be None)
            standardized_outputs = []
            for ei, ism in entry_collection:
                h_var_signature, b_var_signature, h_sat_lit, b_sat_lit \
                    = intermediate_outputs[ei]

                if ism is not None:
                    var_renaming = {
                        t1[0]: t2[0] for t1, t2 in ism["terms"].items()
                    }
                    h_var_signature = [var_renaming[v] for v in h_var_signature]
                    b_var_signature = [var_renaming[v] for v in b_var_signature]
                    h_sat_lit = h_sat_lit.substitute(**ism)
                    b_sat_lit = b_sat_lit.substitute(**ism)

                standardized_outputs.append((
                    h_var_signature, b_var_signature, h_sat_lit, b_sat_lit
                ))

            # Index-neutral flag holding when any (and all) of the explanandum (head(s))
            # in the collection holds
            coll_h_sat_lit = Literal(
                f"coll_head_sat_{i}", wrap_args(*standardized_outputs[0][0])
            )

            # Flags holding when none of the explanantia (bodies) in the collection
            # hold altogether
            coll_h_unexpl_lit = Literal(
                f"coll_head_unexpl_{i}", wrap_args(*standardized_outputs[0][0])
            )
            # Flags holding when each of the standardized explanans (body) holds
            b_expl_lits = [
                Literal(f"coll_head_expl_{i}", wrap_args(*([si]+s_out[0])))
                for si, s_out in enumerate(standardized_outputs)
            ]

            for si, s_out in enumerate(standardized_outputs):
                # coll_h_sat_lit holds when any (and all) of the heads hold
                inference_prog.add_hard_rule(Rule(head=coll_h_sat_lit, body=s_out[2]))

                # Each of b_expl_lits holds if the corresponding b_sat_lit holds
                inference_prog.add_hard_rule(Rule(head=b_expl_lits[si], body=s_out[3]))

            # coll_h_unexpl_lit holds when none of the b_expl_lits hold
            inference_prog.add_hard_rule(Rule(
                head=coll_h_unexpl_lit, body=[xl.flip() for xl in b_expl_lits]
            ))

            # Flag holding when the explanandum (head) is not explained by any of
            # the explanantia (bodies), and thus evoke 'catchall' hypothesis
            r_catchall_lit = Literal(
                f"abduc_catchall_{i}", wrap_args(*standardized_outputs[0][0])
            )

            # r_catchall_lit holds when both coll_h_sat_lit and coll_h_unexpl_lit
            # hold
            inference_prog.add_hard_rule(Rule(
                head=r_catchall_lit, body=[coll_h_sat_lit, coll_h_unexpl_lit]
            ))

            # Compute prior grounded head probabilities, in preparation for weight
            # sum manipulation

            if head_prob != P_C:
                # Manipulate prior prob of grounded body for target conditional,
                # while retaining distributions outside the conditional

                # Manipulating & balancing rule weights
                manipulator = logit(P_C) - logit(1-head_prob)
                balancer = np.log(head_prob) - np.log(1-P_C)

                # manipulator ::  :- not deduc_viol_i_j(*b_var_signature).
                inference_prog.add_rule(
                    Rule(body=grd_r_unsat_lit.flip()), sigmoid(manipulator)
                )
                # balancer ::  :- not head_sat(*h_var_signature).
                inference_prog.add_rule(
                    Rule(body=h_sat_lit.flip()), sigmoid(balancer)
                )
            print(0)
