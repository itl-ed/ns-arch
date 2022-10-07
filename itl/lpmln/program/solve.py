""" Recursive Program().solve() subroutine factored out """
from itertools import product
from functools import reduce
from multiset import FrozenMultiset
from collections import defaultdict

import clingo
import numpy as np
import networkx as nx

from ..literal import Literal
from ..rule import Rule
from ..models import Models
from ..topk_subset import topk_subset_gen
from ..utils import logit


def recursive_solve(prog, topk_ratio, scale_prec):
    """
    Recursively find program solve results, and return as a set of independent
    decision trees. Each child of a node in a tree represents a possible answer
    set for the program, along with the probability value for the model. The
    forest can be used to answer queries on marginal probabilities of atoms.
    Some low-probability models may be pruned, resulting in the total joint
    probabilities not summing to one.
    """
    from .program import Program

    # Can take shortcut if program consists only of grounded facts
    grounded_facts_only = all([
        r.is_fact() and r.is_grounded() for r, _ in prog.rules
    ])
    if grounded_facts_only:
        facts = [
            (
                r.head[0],
                logit(float(r_pr[0]),large="a") if r_pr is not None else None,
                None
            )
            for r, r_pr in prog.rules
        ]

        models = Models(factors=facts)

        return models

    # Feed compiled program string to clingo.Control object and ground program
    rules_obs = _Observer()
    ctl = clingo.Control(["--warn=none"])
    ctl.register_observer(rules_obs)
    ctl.add("base", [], prog._pure_ASP_str())
    ctl.ground([("base", [])])

    # All grounded atoms that are worth considering
    atoms_map = {
        Literal.from_clingo_symbol(atom.symbol): atom.literal
        for atom in ctl.symbolic_atoms
    }
    atoms_inv_map = {v: k for k, v in atoms_map.items()}
    aux_i = max(atoms_map.values()) + 1 if len(atoms_map.values()) > 0 else 0

    # All grounded atoms that each occurring atom can instantiate (grounded atom
    # can instantiate only itself)
    instantiable_atoms = {
        ra: {
            (ma, tuple((rarg[0], marg[0]) for rarg, marg in zip(ra.args, ma.args)))
            for ma in atoms_map
            if ra.name == ma.name and len(ra.args) == len(ma.args) and all([
                rarg[1] == True or rarg[0] == marg[0]
                for rarg, marg in zip(ra.args, ma.args)
            ])
        }
        for ra in prog._rules_by_atom
    }

    # Iterate over the grounded rules for the following processes:
    #   1) Track which rule in the original program could have instantiated each
    #        grounded rule (wish clingo python API supported such feature...)
    #   2) Construct dependency graph from grounded rules
    dep_graph = nx.DiGraph()
    grounded_rules = set()
    grounded_rules_by_head = defaultdict(set)

    for ri, (rule, r_pr) in enumerate(prog.rules):
        # All possible grounded rules that may originate from this rule
        gr_head_insts = [instantiable_atoms[hl.as_atom()] for hl in rule.head]
        gr_head_insts = [
            # Make sure literals (weak-)negated in the original rule are
            # properly flipped
            {(ghl[0].flip(), ghl[1]) if hl.naf else ghl for ghl in ghls}
            for ghls, hl in zip(gr_head_insts, rule.head)
        ]
        gr_body_insts = [instantiable_atoms[bl.as_atom()] for bl in rule.body]
        gr_body_insts = [
            {(gbl[0].flip(), gbl[1]) if bl.naf else gbl for gbl in gbls}
            for gbls, bl in zip(gr_body_insts, rule.body)
        ]

        # Check whether any grounding of this rule would turn out to be never
        # satisfiable because there exists ungroundable positive body atom; in
        # such cases, unsat will never fire for this rule, and we can dismiss
        # this rule altogether without consequences on weight sums
        ungroundable_positive_body_lits = [
            gbls for gbls, bl in zip(gr_body_insts, rule.body)
            if len(gbls)==0 and bl.naf==False
        ]
        if len(ungroundable_positive_body_lits) > 0: continue
        
        # Now get rid of all ungroundable negative body atoms from the body,
        # as they will be trivially checked off in all answer sets
        without_ungroundable_negative_body_lits = [
            gbls for gbls, bl in zip(gr_body_insts, rule.body)
            if bl.naf==False or (len(gbls)>0 and bl.naf==True)
        ]

        # If rule body becomes empty after removing all ungroundable negative
        # body atoms, and there doesn't exist any groundable rule head atom,
        # this rule will be always violated (i.e. unsat will always fire) and
        # can be dismissed as well, but after harvesting weight from the unsat
        groundable_positive_head_lits = [
            ghls for ghls, hl in zip(gr_head_insts, rule.head)
            if len(ghls)>0 and hl.naf==False
        ]
        body_always_sat = len(without_ungroundable_negative_body_lits) == 0
        head_always_unsat = len(groundable_positive_head_lits) == 0
        if body_always_sat and head_always_unsat:
            continue

        # Add all grounded instances of rules that successfully reached here
        # without getting dismissed (unsat weight harvested or not)
        gr_head_insts = list(product(*groundable_positive_head_lits))
        gr_body_insts = list(product(*without_ungroundable_negative_body_lits))
        assert len(gr_head_insts) > 0 and len(gr_body_insts) > 0

        gr_rule_insts_by_body = [
            [
                # 2) Then check for any unifiable rule head+body combinations
                (ghs, gbs) for ghs in gr_head_insts
                if _arg_map_unifiable(ghs+gbs)
            ]
            # 1) First collect by rule body; non-unifiable bodies can be dismissed
            for gbs in gr_body_insts if _arg_map_unifiable(gbs)
        ]
        for insts_per_body in gr_rule_insts_by_body:
            if len(insts_per_body) == 0:
                # Body groundable but no unifiable head groundings that would prevent
                # unsat firing; harvest weight and don't add to grounded_rules
                continue
            else:
                # Both head and body groundable, add to grounded_rules
                grounded_rules |= {
                    (
                        Rule(
                            head=[gh for gh, _ in ghs],
                            body=[gb for gb, _ in gbs]
                        ),
                        r_pr,
                        ri
                    )
                    for ghs, gbs in insts_per_body
                }

    for gr_rule, r_pr, ri in grounded_rules:
        if len(gr_rule.head) > 0:
            for gh in gr_rule.head:
                gh_i = atoms_map[gh.as_atom()]
                grounded_rules_by_head[gh_i].add((gr_rule, r_pr, ri))

                dep_graph.add_node(gh_i)
                for gb in gr_rule.body:
                    gb_i = atoms_map[gb.as_atom()]
                    dep_graph.add_node(gb_i)
                    dep_graph.add_edge(gb_i, gh_i)
        else:
            # Integrity constraint; add rule-specific auxiliary atom
            aux_i += 1
            grounded_rules_by_head[aux_i].add((gr_rule, r_pr, ri))

            aux_lit = Literal("con_aux", args=[(ri, False)])
            atoms_map[aux_lit] = aux_i
            atoms_inv_map[aux_i] = aux_lit

            for gb in gr_rule.body:
                gb_i = atoms_map[gb.as_atom()]
                dep_graph.add_node(gb_i)
                dep_graph.add_node(aux_i)
                dep_graph.add_edge(gb_i, aux_i)

    grounded_rules = FrozenMultiset([(gr_rule, r_pr) for gr_rule, r_pr, _ in grounded_rules])

    # Convert the dependency graph to undirected, and then partition into independent
    # components
    dep_graph_und = dep_graph.to_undirected()
    comps = list(nx.connected_components(dep_graph_und))
    if len(comps) == 1:
        # Graph is connected, with only one component; don't bother to fetch subgraphs
        comps = [dep_graph]
    else:
        comps = [dep_graph.subgraph(nodes) for nodes in comps]
    
    # Grounded rule sets for each component
    grounded_rules_per_comp = [
        set.union(*[grounded_rules_by_head[a] for a in comp.nodes()])
        for comp in comps
    ]

    indep_trees = []
    for ci, comp in enumerate(comps):
        # print(f"A> Let me see... ({ci+1}/{len(comps)})", end="\r")

        # Find possibly relevant rules for each component, ignoring grounded rules that
        # do not overlap at all
        grounded_rules_relevant = grounded_rules_per_comp[ci]

        # Try splitting the program component
        bottom, top, found_split = Program.split(
            comp, atoms_map, grounded_rules_relevant
        )
        is_trivial_split = top is None

        # Check whether the split bottom only consists of body-less grounded rules
        grounded_facts_only = all([
            r.is_fact() and r.is_grounded() for r, _ in bottom.rules
        ])

        if is_trivial_split:
            # Empty program top (i.e. recursion base case)
            if grounded_facts_only:
                # If the bottom (== comp) consists only of grounded choices without rule
                # body, may directly return the factored representation of individual
                # probabilistic choices -- no pruning needed
                facts = [
                    prog._rules_by_atom[atoms_inv_map[v]] for v in comp.nodes
                    if atoms_inv_map[v] in prog._rules_by_atom
                ]

                # (Assuming only one fact is present with the atom as head)
                facts = [prog.rules[fs.pop()] for fs in facts]
                facts = [
                    (
                        f.head[0],
                        logit(float(r_pr[0]), large="a")
                            if r_pr is not None else None,
                        None
                    )
                    for f, r_pr in facts
                ]

                tree = Models(factors=facts)

            else:
                # Solve the bottom (== comp) with clingo and return answer sets (with
                # associated probabilities)
                ctl = clingo.Control(["--warn=none"])
                ctl.add("base", [], bottom._pure_ASP_str(unsats=True))
                ctl.ground([("base", [])])
                ctl.configuration.solve.models = 0

                models = []
                with ctl.solve(yield_=True) as solve_gen:
                    for m in solve_gen:
                        models.append(m.symbols(atoms=True))
                        if solve_gen.get().unsatisfiable: break

                # Process models to extract true atoms along with model weights
                models = [
                    ([a for a in m if a.name != "unsat"], [a for a in m if a.name == "unsat"])
                    for m in models if len(m) > 0
                ]
                models = [
                    (
                        [Literal.from_clingo_symbol(a) for a in atoms],
                        reduce(_sum_weights, [
                            ([0.0, -1.0] if a.arguments[1].positive else [0.0, 1.0])
                            if a.arguments[1].type == clingo.SymbolType.Function
                            else [-a.arguments[1].number / scale_prec, 0.0] for a in unsats
                        ])
                    )
                    for atoms, unsats in models
                ]

                if len(models) > 0:
                    # Clear models instance to be fed as top_models. Represents Models
                    # covering an empty model (caution: instance itself is not empty!)
                    models_clear = Models(factors=[])

                    outcomes = [
                        (atoms, weight, models_clear, False)
                        for atoms, weight in models
                    ]
                    tree = Models(outcomes=outcomes)
                else:
                    tree = Models()        # Empty models

        else:
            # Solve the bottom for possible models with probabilities, obtain & solve the
            # reduced top for each model
            if grounded_facts_only:
                # If program only consists of grounded choice facts, may bypass clingo and
                # find models along with probabilities combinatorially, possibly pruning
                # low-probability models from the bottom.
                # (Cannot use factored representation since we need to reduce program top
                # for each possible model of the program bottom.)
                bottom_models = []
                
                # Only need to consider soft rules (i.e. rules with 0.0 < r_pr < 1.0) when
                # finding top-k models with this method
                abs_facts = [rule for rule, r_pr in bottom.rules if r_pr is None]
                abs_facts = {rule.head[0] for rule in abs_facts}

                incid_facts = [(rule, r_pr) for rule, r_pr in bottom.rules if r_pr is not None]
                soft_facts = [(rule, r_pr) for rule, r_pr in incid_facts if 0.0 < r_pr[0] < 1.0]
                hard_facts = [
                    (rule, r_pr) for rule, r_pr in incid_facts if r_pr[0] == 0.0 or r_pr[0] == 1.0
                ]

                if len(soft_facts) > 0:
                    # Aggregate rules with same head atom, combining weights
                    soft_facts_agg = defaultdict(float)
                    for rule, r_pr in soft_facts:
                        soft_facts_agg[rule.head[0]] += logit(r_pr[0])
                    soft_facts = [(Rule(head=head), w) for head, w in soft_facts_agg.items()]

                    # Rules should be sorted by weights first to apply the algorithm
                    soft_facts = sorted(soft_facts, key=lambda rw: rw[1], reverse=True)

                    # Using logits of r_pr values as rule weights ensures direct association of
                    # the rule weights with the marginal probabilities of rule head atoms across
                    # all possible models (... on the assumption that there are no probabilistic
                    # choice rules with the same head atoms with non-disjoint body in program)
                    rule_weights = [rw for _, rw in soft_facts]

                    # (Log of) partition function for all the soft rules can be analytically
                    # computed as below
                    logZ = sum([np.log(1+np.exp(w)) for w in rule_weights])

                    # Log of total probability mass covered, from the top; need to query models
                    # until more than aggregate probability mass gets larger than top_k
                    log_pmass_covered = float("-inf")      # Represents limit(log(x)) as x -> +0

                    # Collect most probable possible worlds
                    subsets = []
                    subset_generator = topk_subset_gen(rule_weights)
                    for subset, weight_sum in subset_generator:
                        # Update pmass_covered with log-sum-exp
                        log_joint_p = weight_sum - logZ
                        log_pmass_covered = np.logaddexp(log_pmass_covered, log_joint_p)

                        # Append model retrieved from the indices in subset along with weight
                        # sum
                        subsets.append((subset, np.exp(log_joint_p)))
                        bottom_models.append((
                            {sr.head[0] for i, (sr, _) in enumerate(soft_facts) if i in subset},
                            {sr.head[0] for i, (sr, _) in enumerate(soft_facts) if i not in subset},
                            [weight_sum, 0.0]
                        ))
                        
                        # Break with sufficient coverage
                        if log_pmass_covered >= np.log(topk_ratio):
                            break
                    subset_generator.close()
                
                # Add hard-weighted facts
                if len(hard_facts) > 0:
                    if len(bottom_models) == 0:
                        # Need to add an empty model if no models in list at this point
                        bottom_models.append((set(), set(), [0.0, 0.0]))
                    hard_pos_atoms = {hr.head[0] for hr, r_pr in hard_facts if r_pr[0]==1.0}
                    hard_neg_atoms = {hr.head[0] for hr, r_pr in hard_facts if r_pr[0]==0.0}
                    bottom_models = [
                        (
                            pos_atoms | hard_pos_atoms,
                            neg_atoms | hard_neg_atoms,
                            _sum_weights(weight_sum, [0, len(hard_pos_atoms)+len(hard_neg_atoms)])
                        )
                        for pos_atoms, neg_atoms, weight_sum in bottom_models
                    ]
                
                # Add absolute, definitionally derived facts
                if len(abs_facts) > 0:
                    if len(bottom_models) == 0:
                        # Need to add an empty model if no models in list at this point
                        bottom_models.append((set(), set(), [0.0, 0.0]))
                    bottom_models = [
                        (pos_atoms | abs_facts, neg_atoms, weight_sum)
                        for pos_atoms, neg_atoms, weight_sum in bottom_models
                    ]
            else:
                # Solve the bottom (== comp) with clingo and return answer sets (with
                # associated probabilities)
                bottom_atoms = {atoms_inv_map[n] for n in found_split}

                ctl = clingo.Control(["--warn=none"])
                ctl.add("base", [], bottom._pure_ASP_str(unsats=True))
                ctl.ground([("base", [])])
                ctl.configuration.solve.models = 0

                bottom_models = []
                with ctl.solve(yield_=True) as solve_gen:
                    for m in solve_gen:
                        bottom_models.append(m.symbols(atoms=True))
                        if solve_gen.get().unsatisfiable: break

                # Process models to extract true atoms along with model weights
                bottom_models = [
                    ([a for a in m if a.name != "unsat"], [a for a in m if a.name == "unsat"])
                    for m in bottom_models
                ]
                bottom_models = [
                    (
                        [Literal.from_clingo_symbol(a) for a in atoms],
                        reduce(_sum_weights, [
                            ([0.0, -1.0] if a.arguments[1].positive else [0.0, 1.0])
                            if a.arguments[1].type == clingo.SymbolType.Function
                            else [-a.arguments[1].number / scale_prec, 0.0] for a in unsats
                        ])
                    )
                    for atoms, unsats in bottom_models
                ]

                if len(bottom_models) > 0:
                    # TODO: Not updated to match the updated solving procedure, update
                    raise NotImplementedError

                    logZ = reduce(np.logaddexp, [weight for _, weight in bottom_models])
                    bottom_models = [
                        (set(atoms), np.exp(weight-logZ)) for atoms, weight in bottom_models
                    ]
                    bottom_models = [
                        # Positive atoms, negative atoms, probability
                        (model, bottom_atoms-model, pr) for model, pr in bottom_models
                    ]

            # Solve reduced program top for each discovered model; first compute program
            # reduction by the common atoms
            atom_sets = [
                {(pl, True) for pl in bm[0]} | {(nl, False) for nl in bm[1]}
                for bm in bottom_models
            ]
            atom_commons = set.intersection(*atom_sets) if len(atom_sets) > 0 else set()
            reduced_common, base_w_reduc_com = top.reduce(
                {atm for atm, pos in atom_commons if pos},
                {atm for atm, pos in atom_commons if not pos}
            )

            # Now for each bottom model reduce the common reduction with the remainder of
            # the atoms, and solve the fully reduced
            outcomes = []
            atom_diffs = [a-atom_commons for a in atom_sets]
            for bi, ((pos_atoms, _, w), atoms) in enumerate(zip(bottom_models, atom_diffs)):
                # print(f"A> Let me see... ({bi+1}/{len(bottom_models)})", end="\r")

                pos_atoms_diff = {atm for atm, pos in atoms if pos}
                neg_atoms_diff = {atm for atm, pos in atoms if not pos}

                reduced_top, base_w_reduc_top = reduced_common.reduce(
                    pos_atoms_diff, neg_atoms_diff
                )
                top_models = recursive_solve(
                    reduced_top, topk_ratio, scale_prec
                )

                final_w = reduce(_sum_weights, [w, base_w_reduc_com, base_w_reduc_top])
                outcomes.append((pos_atoms, final_w, top_models, False))
            tree = Models(outcomes=outcomes)

        indep_trees.append(tree)
        # print("A>" + (" "*50), end="\r")

    models = Models(factors=indep_trees)

    return models


class _Observer:
    """ For tracking added grounded rules """
    def __init__(self):
        self.rules = []
    def rule(self, choice, head, body):
        self.rules.append((head, body, choice))


def _arg_map_unifiable(g_lits):
    """ Test if all argument mappings are unifiable """
    arg_maps = sum([map for _, map in g_lits], ())
    mapping = {}
    for ra, ma in arg_maps:
        if ra in mapping:
            if mapping[ra] != ma:
                return False
        else:
            mapping[ra] = ma

    return True


def _sum_weights(w1, w2):
    """
    Sum of weight sums, each represented as a binomial consisting of a zeroth
    degree term and first degree term (for "a"), represented as iterable of
    number with length of 2
    """
    assert len(w1) == 2 and len(w2) == 2
    return [w1[0]+w2[0], w1[1]+w2[1]]
