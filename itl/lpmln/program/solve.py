""" Recursive Program().solve() subroutine factored out """
import operator
from itertools import product, chain, combinations
from functools import reduce
from multiset import FrozenMultiset
from collections import defaultdict

import clingo
import networkx as nx

from ..literal import Literal
from ..rule import Rule
from ..polynomial import Polynomial
from ..models import Models, ModelsCommonFactors, ModelsBranchOutcomes
from ..utils import logit
from ..utils.logic import cnf_to_dnf, simplify_cnf


def recursive_solve(prog, scale_prec):
    """
    Recursively find program solve results, and return as a set of independent
    decision trees. Each child of a node in a tree represents a possible answer
    set for the program, along with the probability value for the model. The
    forest can be used to answer queries on marginal probabilities of atoms.
    """
    from .program import Program

    # Can take shortcut if program consists only of grounded facts
    if _grounded_facts_only([r for r, _ in prog.rules]):
        return _models_from_rule_heads(prog.rules)

    grounded_rules_by_head, comps, atoms_map, atoms_inv_map = _ground_and_factorize(prog)

    indep_trees = []
    for comp in comps:
        # If subprogram corresponding to this component only consists of grounded facts,
        # things get much easier
        gr_comp_prog = set.union(*[grounded_rules_by_head[a] for a in comp])

        if _grounded_facts_only([r for r, _, _ in gr_comp_prog]):
            # Directly return the factored representation of individual probabilistic
            # choices

            # (Assuming only one fact is present with the atom as head)
            facts = [
                prog._rules_by_atom[atoms_inv_map[v]] for v in comp.nodes
                if atoms_inv_map[v] in prog._rules_by_atom
            ]
            facts = [prog.rules[list(fs)[0]] for fs in facts]

            comp_models = _models_from_rule_heads(facts)
            indep_trees.append(comp_models)
            continue

        # Otherwise, need proper attempt to solve the subprogram - see it how factorizes
        # for more efficient solving and treat accordingly. The code below implements an
        # algorithm for finding best factorization of program solving process, which would
        # lead to minimal total amount of computation required for this component; for
        # optimizing solution of combinatorially demanding programs

        # Find condensation of the component where strongly connected components are
        # reduced into single vertices (if component is already DAG, condensation will
        # be topologically equivalent to original component)
        comp_condensed = nx.condensation(comp)
        scc_members = nx.get_node_attributes(comp_condensed, "members")

        # Locate all sink nodes in the condensed component
        sinks = [n for n, deg in comp_condensed.out_degree if deg==0]

        # Initialize frontier as singleton list containing a dummy node. The frontier is
        # defined at the condensation level, where each node corresponds to independently
        # solvable subprograms given values of condensation-parent nodes.
        comp_condensed.add_node(-1)
        for sn in sinks:
            comp_condensed.add_edge(sn, -1)
        frontier = {frozenset([-1])}

        # Then continue expanding the frontier downward by replacing each node with its
        # parent nodes until source nodes are reached. During frontier expansion, mind
        # overlapping parents.
        unexplored_frontier = True; reached_source = set(); backtrack_paths = {}
        while unexplored_frontier:
            expanded = False; new_frontier_nodes = set()
            for fns in frontier:
                # Check for other nodes with overlapping parents
                fns_parents = frozenset([
                    parent for fn in fns for parent, _ in comp_condensed.in_edges(fn)
                    if parent not in fns    # Safely disregard dependency within fns
                ])
                if len(fns_parents) == 0:
                    # No parent nodes at all
                    reached_source.add(fns)
                    continue

                overlapping_parents = {
                    cns: fns_parents & frozenset.union(*pnss)
                    for cns, pnss in backtrack_paths.items()
                }
                overlapping_parents = {
                    cns: pns for cns, pns in overlapping_parents.items()
                    if len(pns) > 0
                }

                # Merge any overlaps to form coupled node sets, effectively partitioning
                # the parent nodes
                overlap_chunks = set()
                for overlap in overlapping_parents.values():
                    chunks_to_merge = {
                        chk for chk in overlap_chunks if len(chk & overlap) > 0
                    }

                    if len(chunks_to_merge) == 0:
                        # Fresh chunk to add
                        overlap_chunks.add(overlap)
                    else:
                        # Replace chunks with overlaps with a new merged chunk
                        merged_chunk = frozenset.union(*chunks_to_merge)
                        overlap_chunks.add(merged_chunk)
                        for chk in chunks_to_merge: overlap_chunks.remove(chk)
                
                # Finalized partitioning of parents
                coupled_parents = frozenset.union(*overlap_chunks) \
                    if len(overlap_chunks) > 0 else frozenset()
                unseen_parents = fns_parents - coupled_parents

                parent_node_sets = {frozenset([pn]) for pn in unseen_parents}
                parent_node_sets |= overlap_chunks

                backtrack_paths[fns] = parent_node_sets

                if len(unseen_parents) > 0:
                    new_frontier_nodes |= set(parent_node_sets)
                    expanded = True

            # Update frontier and test if further (tests for) expansions are necessary
            unexplored_frontier = expanded
            if unexplored_frontier:
                frontier = new_frontier_nodes
        del backtrack_paths[frozenset([-1])]    # Dummy not needed, shouldn't be iterated over

        # Set of all chunks where any overlapping ones are merged
        chunks_witnessed = reached_source | set(backtrack_paths)
        if len(backtrack_paths) > 0:
            chunks_witnessed |= set.union(*backtrack_paths.values())
        
        all_chunks = set()
        for cns in chunks_witnessed:
            if any(cns > chk for chk in all_chunks):
                all_chunks = {cns if cns > f else f for f in all_chunks}
            elif any(cns < chk for chk in all_chunks):
                continue
            else:
                all_chunks.add(cns)
        
        # Finalized frontier
        frontier_final = set()
        for cns in reached_source:
            supersets = {chk for chk in all_chunks if cns < chk}
            if len(supersets) > 0:
                frontier_final |= supersets
            else:
                frontier_final.add(cns)

        if len(backtrack_paths) > 0:
            # Update backtrack_paths so that it complies with the partitioning of parent
            # node sets in frontier_final
            for cns, pnss in backtrack_paths.items():
                matching_chunks = [
                    [f for f in frontier_final if pns < f] for pns in pnss
                ]
                backtrack_paths[cns] = {
                    chks[0] if len(chks) > 0 else pns
                    for pns, chks in zip(pnss, matching_chunks)
                }
            
            # and vice versa


        # For each node set in the finalized frontier, find the subgraph and then (grounded)
        # subprogram such that estimation of probabilities for the node set can be obtained
        # by solving the subprogram
        models_per_node_set = {}
        for fns in frontier_final:
            # Appropriate subgraph for a frontier node set is defined by the set of ancestor
            # nodes and self
            ancestors_and_me = set.union(*[
                nx.ancestors(comp_condensed, fn) | {fn} for fn in fns
            ])
            subgraph = comp_condensed.subgraph(ancestors_and_me)

            # Extract the grounded subprogram to solve that corresponds to the subgraph;
            # first recover set of original component nodes from the condensed subgraph
            # nodes, then use it to synthesize the desired subprogram
            orig_nodes = set.union(*[mems for _, mems in subgraph.nodes(data="members")])
            gr_subprog = set.union(*[grounded_rules_by_head[a] for a in orig_nodes])
            subcomp = comp.subgraph(orig_nodes)

            # Solve this subprogram; how it will be achieved depends on the result of
            # splitting this subprogram
            bottom, top, _ = Program.split(subcomp, atoms_map, gr_subprog)
            bottom_gr_facts_only = _grounded_facts_only([r for r, _ in bottom.rules])
            is_trivial_split = top is None

            if is_trivial_split:
                # Empty program top (i.e. recursion base case)
                if bottom_gr_facts_only:
                    # (Assuming only one fact is present with the atom as head)
                    subprog_models = _models_from_rule_heads(bottom.rules)
                else:
                    # Solve the bottom with clingo and return answer sets (with total
                    # associated total weights)
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
                            {Literal.from_clingo_symbol(a) for a in atoms},
                            reduce(operator.mul, [
                                (Polynomial(terms={ -1: 0.0 }) if a.arguments[1].positive
                                    else Polynomial(terms={ 1: 0.0 }))
                                if a.arguments[1].type == clingo.SymbolType.Function
                                else Polynomial(terms={ 0: -a.arguments[1].number / scale_prec })
                                for a in unsats
                            ])
                        )
                        for atoms, unsats in bottom_models
                    ]

                    all_atoms = set.union(*[bm for bm, _ in bottom_models])
                    bottom_outcomes = [
                        (
                            bm | {atm.flip() for atm in all_atoms-bm}, None, weights_exp,
                            Polynomial(float_val=1.0), None, False
                        )
                        for bm, weights_exp in bottom_models
                    ]
                    subprog_models = Models([
                        (None, ModelsBranchOutcomes(bottom_outcomes))
                    ])
            else:
                # Solve the bottom for models, obtain & solve the reduced top for
                # each bottom model
                if bottom_gr_facts_only:
                    bottom_factors = [
                        Models(factors=[(
                            f.head[0] if len(f.head)>0 else None,
                            logit(float(r_pr[0]), large="a")
                                if r_pr is not None else None,
                            None
                        )])
                        for f, r_pr in bottom.rules
                    ]
                    bottom_nodes_covered = [{br[0].head[0]} for br in bottom.rules]

                    subprog_models = _models_from_bottom_factors_and_top(
                        bottom_factors, bottom_nodes_covered, top, scale_prec
                    )
                else:
                    raise NotImplementedError

            models_per_node_set[fns] = subprog_models

        # Based on the results for the frontier node sets, build final results in a
        # bottom-up manner, until all atoms covered by this original component are
        # accounted for
        while len(backtrack_paths) > 0:
            cnss_to_process = [
                cns for cns, pnss in backtrack_paths.items()
                if all(pns in models_per_node_set for pns in pnss)
            ]
            for cns in cnss_to_process:
                cns_nodes_covered = set.union(*[scc_members[cn] for cn in cns])

                # Fetch relevant grounded rules, attach handles to respective rule bodies
                # (dividing positive/negative parts)
                relevant_rules = [
                    gr[:2] for n in cns_nodes_covered for gr in grounded_rules_by_head[n]
                ]

                # Collect models for subprograms corresponding to the parents of this node,
                # enumerate possible outcomes abstracted with respect to parts of bodies of
                # relevant rules, gathering unnormalized probability masses for each branch
                # along with common partition function Z
                cns_parent_sets = backtrack_paths.pop(cns)

                bottom_factors = [models_per_node_set[pns] for pns in cns_parent_sets]
                bottom_nodes_covered = [
                    {atoms_inv_map[n] for n in set.union(*[scc_members[pn] for pn in pns])}
                    for pns in cns_parent_sets
                ]
                top_program = Program(relevant_rules)

                models_per_node_set[cns] = _models_from_bottom_factors_and_top(
                    bottom_factors, bottom_nodes_covered, top_program, scale_prec
                )

        # By now all sink nodes in the condensed component should have been solved
        # and assigned some Models instance. Construct a Models instance as the final
        # representation of probabilistic answer sets engendered by this component.
        sink_nodes_models = [
            ms for cns, ms in models_per_node_set.items()
            if all(cn in sinks for cn in cns)
        ]
        comp_cfs_bos_pairs = sum([ms.cfs_bos_pairs for ms in sink_nodes_models], [])
        comp_models = Models(comp_cfs_bos_pairs)
        indep_trees.append(comp_models)

    models = Models([(ModelsCommonFactors(indep_trees), None)])

    return models


class _Observer:
    """ For tracking added grounded rules """
    def __init__(self):
        self.rules = []
    def rule(self, choice, head, body):
        self.rules.append((head, body, choice))


def _ground_and_factorize(prog):
    """
    Ground program, construct a directed graph reflecting dependency between grounded
    atoms (by program rules), find independent graph component factors. Return the
    grounded rules indexed by their heads, component graphs, and mappings between
    grounded atoms and their integer indices.
    """
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

        # Possible mappings from variables to constants worth considering
        if len(gr_head_insts+gr_body_insts) > 0:
            possible_substs = set.union(*[
                set.union(*[set(gl[1]) for gl in gls]) if len(gls)>0 else set()
                for gls in gr_head_insts+gr_body_insts
            ])          # All var-cons pair witnessed
            possible_substs = {
                t1_1: {t2_2 for t1_2, t2_2 in possible_substs if t1_1==t1_2}
                for t1_1, _ in possible_substs
            }           # Collect by variable
            possible_substs = [
                # t1!=t2 ensures t1 is a variable
                {t1: t2 for t1, t2 in zip(possible_substs, cs) if t1!=t2}
                for cs in product(*possible_substs.values())
            ]           # Flatten products into list of all possible groundings
        else:
            possible_substs = [{}]
        
        for subst in possible_substs:
            # For each possible grounding of this rule
            subst = { (v, True): (c, False) for v, c in subst.items() }
            gr_rule = rule.substitute(terms=subst)

            gr_body_pos = [gbl for gbl in gr_rule.body if gbl.naf==False]
            gr_body_neg = [gbl for gbl in gr_rule.body if gbl.naf==True]

            # Check whether this grounded rule would turn out to be never satisfiable
            # because there exists ungroundable positive body atom; in such cases,
            # unsat will never fire, and we can dismiss the rule
            if any(gbl not in atoms_map for gbl in gr_body_pos):
                continue

            # Negative rule body after dismissing ungroundable atoms; ungroundable
            # atoms in negative body can be ignored as they are trivially satisfied
            # (always reduced as all models will not have occurrence of the atom)
            gr_body_neg_filtered = [
                gbl for gbl in gr_body_neg if gbl.as_atom() in atoms_map
            ]
            gr_body_filtered = gr_body_pos + gr_body_neg_filtered

            # Add this grounded rule to the list with r_pr and index
            gr_rule = Rule(head=gr_rule.head, body=gr_body_filtered)
            grounded_rules.add((gr_rule, r_pr, ri))

    for gr_rule, r_pr, ri in grounded_rules:
        if len(gr_rule.head) > 0:
            for ghl in gr_rule.head:
                if ghl.as_atom() not in atoms_map:
                    aux_i += 1
                    atoms_map[ghl.as_atom()] = aux_i
                    atoms_inv_map[aux_i] = ghl.as_atom()

                ghl_i = atoms_map[ghl.as_atom()]
                grounded_rules_by_head[ghl_i].add((gr_rule, r_pr, ri))

                dep_graph.add_node(ghl_i)
                for gbl in gr_rule.body:
                    gbl_i = atoms_map[gbl.as_atom()]
                    dep_graph.add_node(gbl_i)
                    dep_graph.add_edge(gbl_i, ghl_i)
        else:
            # Integrity constraint; add rule-specific auxiliary atom
            aux_i += 1
            grounded_rules_by_head[aux_i].add((gr_rule, r_pr, ri))

            aux_lit = Literal("con_aux", args=[(ri, False)])
            atoms_map[aux_lit] = aux_i
            atoms_inv_map[aux_i] = aux_lit

            dep_graph.add_node(aux_i)
            for gbl in gr_rule.body:
                gbl_i = atoms_map[gbl.as_atom()]
                dep_graph.add_node(gbl_i)
                dep_graph.add_edge(gbl_i, aux_i)

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

    return grounded_rules_by_head, comps, atoms_map, atoms_inv_map


def _grounded_facts_only(rules):
    """ Test if set of rules consists only of grounded facts """
    return all(r.is_fact() and r.is_grounded() for r in rules)


def _models_from_rule_heads(rules):
    """
    Compose a factors-Models instance from set of (grounded) rules using their heads
    """
    rules_factors = [
        (
            f.head[0] if len(f.head)>0 else None,
            logit(float(r_pr[0]), large="a")
                if r_pr is not None else None,
            None
        )
        for f, r_pr in rules
    ]
    return Models([
        (ModelsCommonFactors(rules_factors), None)
    ])


def _models_from_bottom_factors_and_top(
        bottom_factors, bottom_nodes_covered_per_factor, top_program, scale_prec
    ):
    bottom_indep_branches = []
    top_rule_bodies = [
        (
            {gbl for gbl in gr.body if gbl.naf==False},
            {gbl for gbl in gr.body if gbl.naf==True}
        )
        for gr, _ in top_program.rules
    ]
    for factor, nodes_covered in zip(bottom_factors, bottom_nodes_covered_per_factor):
        assert isinstance(factor, Models)
        
        # For each relevant top rule found, prepare representation of the event
        # that corresponding body parts of the rule are satisfied, as a formula
        # in conjunctive normal form (list of sets here); will be used to filter
        # cached_models for each possible outcome branch later
        filter_events = [
            (
                body_pos & nodes_covered,
                body_neg & {gl.flip() for gl in nodes_covered}
            )
            for body_pos, body_neg in top_rule_bodies
        ]
        filter_events = [
            (
                [{gl} for gl in body_pos | body_neg],           # Body sat
                [{gl.flip() for gl in body_pos | body_neg}]     # Body unsat
            )
            for body_pos, body_neg in filter_events
        ]
        relevant_influx_rules = {
            i for i, (ev_conj_sat, _) in enumerate(filter_events)
            if len(ev_conj_sat) > 0
        }

        # All possible outcome branches with respect to whether each relevant
        # rule body parts are satisfied or not, along with indices of rules for
        # which corresponding body parts are satisfied
        branches = []
        for rule_inds in _powerset(relevant_influx_rules):
            branch = []
            for ri in rule_inds:
                ev_conj_sat, _ = filter_events[ri]
                branch += ev_conj_sat
            for ri in relevant_influx_rules-set(rule_inds):
                _, ev_conj_unsat = filter_events[ri]
                branch += ev_conj_unsat
            branch = simplify_cnf(branch)

            if all(len(cjct) > 0 for cjct in branch):
                # If any conjunct is an empty disjunction, this branch event is
                # logically impossible
                branches.append(branch)

        bottom_indep_branches.append(branches)

    # Package into a common factors instance
    bottom_factors = ModelsCommonFactors(bottom_factors)

    # Process each full specification of independent branch cases and
    # aggregate total (unnormalized) probability mass
    consequences_per_choice = []
    for branch_choice in product(*bottom_indep_branches):
        total_ev_conj = sum([bc for bc in branch_choice], [])

        # Converting CNF to DNF, so that we can process each possible
        # disjunct case (each disjunct, which is conjunction of atoms,
        # will be used to reduce top program)
        total_ev_disj = cnf_to_dnf(total_ev_conj)

        # For each disjunct, filter bottom_factors with corresponding event and
        # obtain unnormalized probability for the branch
        bottom_factors_filtered = [
            reduce(
                lambda bf, ls: bf.filter(ls),
                [bottom_factors]+[frozenset([lit]) for lit in djct]
            )
            for djct in total_ev_disj
        ]
        unnorm_pmasses = [
            bf_filtered.compute_Z()
            for bf_filtered in bottom_factors_filtered
        ]

        # Weed out branch with zero possibility (i.e. zero probability mass)
        total_ev_disj = [
            (djct, bf_filtered, pmass)
            for djct, bf_filtered, pmass
            in zip(total_ev_disj, bottom_factors_filtered, unnorm_pmasses)
            if not pmass.is_zero()
        ]

        reduced_top_programs = [
            top_program.reduce(djct) for djct, _, _ in total_ev_disj
        ]

        # Add results of reducing top program with the branch event(s) and
        # solving to list of consequences per branch event
        consequences_per_choice += [
            (
                djct, bf_filtered, pmass, base_w,
                recursive_solve(reduced_prog, scale_prec), False
            )
            for (djct, bf_filtered, pmass), (reduced_prog, base_w)
            in zip(total_ev_disj, reduced_top_programs)
        ]

    # Collate the result into an outcomes-Models instance
    cfs_bos_pairs = [(bottom_factors, ModelsBranchOutcomes(consequences_per_choice))]
    return Models(cfs_bos_pairs)


def _powerset(iterable):
    """ Set of all subsets of iterable """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
