""" Program().compile() subroutine factored out """
import operator
from functools import reduce
from itertools import product, combinations
from collections import defaultdict

import clingo
import networkx as nx

from ..literal import Literal
from ..rule import Rule
from ..polynomial import Polynomial
from ..utils import logit


def compile(prog):
    """
    Compiles program into a binary join tree from equivalent directed graph, which
    would contain data needed to answer probabilistic inference queries. Achieved
    by running the following operations:

    1) Ground the program with clingo to find set of actually grounded atoms
    2) Compile the factor graph into binary join tree, following the procedure
        presented in Shenoy, 1997 (modified to comply with semantics of logic
        programs). Include singleton node sets for each grounded atom, so that
        the compiled join tree will be prepared to answer marginal queries for
        atoms afterwards.
    3) Identify factor potentials associated with valid sets of nodes. Each
        program rule corresponds to a piece of information contributing to the
        final factor potential for each node set. Essentially, this amounts to
        finding a factor graph corresponding to the provided program.
    4) Run modified Shafer-Shenoy belief propagation algorithm to fill in values
        needed for answering probabilistic inference queries later.
    
    Returns a binary join tree populated with the required values resulting from belief
    propagation.
    """
    # Construct a binary join tree that contains all procedural information and data
    # needed for belief propagation
    bjt = _construct_BJT(prog)

    if bjt is not None:
        # Run modified Shafer-Shenoy belief propagation on the constructed binary
        # join tree, filling in the output (unnormalized) belief storage registers
        output_node_sets = [frozenset({atm}) for atm in bjt.graph["atoms_map_inv"]]

        for node_set in output_node_sets:
            _belief_propagation(bjt, node_set)

    return bjt


def bjt_query(bjt, q_key):
    """ Query a BJT for (unnormalized) belief table """
    relevant_nodes = frozenset({abs(n) for n in q_key})
    relevant_cliques = [n for n in bjt.nodes if relevant_nodes <= n]

    if len(relevant_cliques) > 0:
        # In-clique query; find the BJT node with the smallest node set that
        # comply with the key
        smallest_rel_nodeset = sorted(relevant_cliques, key=len)[0]
        _belief_propagation(bjt, smallest_rel_nodeset)
        beliefs = bjt.nodes[smallest_rel_nodeset]["output_beliefs"]

        # Marginalize and return
        return _marginalize_simple(beliefs, relevant_nodes)
    else:
        # Out-clique query; first divide query key node set by belonging components
        # in the BJT
        components = {
            frozenset.union(*comp): bjt.subgraph(comp)
            for comp in nx.connected_components(bjt.to_undirected())
        }

        divided_keys_and_subtrees = {
            frozenset({l for l in q_key if abs(l) in comp_nodes}): sub_bjt
            for comp_nodes, sub_bjt in components.items()
            if len(comp_nodes & relevant_nodes) > 0
        }

        if len(divided_keys_and_subtrees) == 1:
            # All query key nodes in the same component; variable elimination needed
            raise NotImplementedError
        else:
            # Recursively query each subtree with corresponding 'subkey'
            query_results = {
                subkey: bjt_query(sub_bjt, subkey)
                for subkey, sub_bjt in divided_keys_and_subtrees.items()
            }

            # Combine independent query results
            return _combine_factors_simple(list(query_results.values()))


class _Observer:
    """ For tracking added grounded rules """
    def __init__(self):
        self.rules = []
    def rule(self, choice, head, body):
        self.rules.append((head, body, choice))


def _construct_BJT(prog):
    bjt = nx.DiGraph()

    if _grounded_facts_only([r for r, _ in prog.rules]):
        # Can take simpler shortcut for constructing binary join tree if program
        # consists only of grounded facts
        bjt.graph["atoms_map"] = {}
        bjt.graph["atoms_map_inv"] = {}

        for i, (rule, r_pr) in enumerate(prog.rules):
            # Integer indexing should start from 1, to represent negated atoms as
            # negative ints
            i += 1

            # Mapping between atoms and their integer indices
            bjt.graph["atoms_map"][rule.head[0]] = i
            bjt.graph["atoms_map_inv"][i] = rule.head[0]

            fact_input_potential = _rule_to_potential(rule, r_pr, { rule.head[0]: i })

            # Add singleton atom set node for the atom, with an appropriate input
            # potential from the rule weight
            bjt.add_node(frozenset({i}), input_potential=fact_input_potential)

    else:
        grounded_rules_by_atms, atoms_map, atoms_map_inv = _ground_and_index(prog)

        # Mapping between atoms and their integer indices
        bjt.graph["atoms_map"] = atoms_map
        bjt.graph["atoms_map_inv"] = atoms_map_inv

        if len(atoms_map) == 0 or len(grounded_rules_by_atms) == 0:
            # Happens to have no atoms and rules grounded; nothing to do here
            pass
        else:
            # Binary join tree construction (cf. Shenoy, 1997)
            node_sets_unpr = set(atoms_map_inv)
            node_set_sets_unpr = set(grounded_rules_by_atms)
            node_set_sets_unpr |= {frozenset({atm}) for atm in atoms_map_inv}

            while len(node_set_sets_unpr) > 1:
                # Pick a variable with a heuristic, namely one that would lead to smallest
                # union of relevant nodes in factors
                node_set_unions = {
                    n: frozenset.union(*{ns for ns in node_set_sets_unpr if n in ns})
                    for n in node_sets_unpr
                }
                node = sorted(node_sets_unpr, key=lambda n: len(node_set_unions[n]))[0]
                relevant_node_sets = {ns for ns in node_set_sets_unpr if node in ns}

                while len(relevant_node_sets) > 1:
                    # Pick a node set pair that would give smallest union
                    node_set_pair_unions = {
                        (ns1, ns2): ns1 | ns2
                        for ns1, ns2 in combinations(relevant_node_sets, 2)
                    }
                    ns1, ns2 = sorted(
                        node_set_pair_unions, key=lambda nsp: len(node_set_pair_unions[nsp])
                    )[0]
                    ns_union = ns1 | ns2

                    # Add (possibly new) nodes and edges (both directions, for storing messages
                    # of two opposite directions)
                    bjt.add_node(ns1); bjt.add_node(ns2); bjt.add_node(ns_union)
                    if ns1 != ns_union:
                        bjt.add_edge(ns1, ns_union); bjt.add_edge(ns_union, ns1)
                    if ns2 != ns_union:
                        bjt.add_edge(ns2, ns_union); bjt.add_edge(ns_union, ns2)

                    # Update relevant node set set; # element reduced exactly by one
                    relevant_node_sets -= {ns1, ns2}
                    relevant_node_sets |= {ns_union}

                if len(node_sets_unpr) > 1:
                    node_set = list(relevant_node_sets)[0]
                    node_set_compl_node = node_set - {node}

                    bjt.add_node(node_set)
                    if len(node_set_compl_node) > 0:
                        # Should process empty set only; usually (I think) empty
                        # node_set_compl_node signals disconnected components
                        # Add (possibly new) nodes and edges
                        bjt.add_node(node_set_compl_node)
                        if node_set != node_set_compl_node:
                            bjt.add_edge(node_set, node_set_compl_node)
                            bjt.add_edge(node_set_compl_node, node_set)

                        # Update node_sets_unpr and node_set_sets_unpr
                        node_set_sets_unpr |= {node_set_compl_node}

                    node_set_sets_unpr -= {ns for ns in node_set_sets_unpr if node in ns}
                    node_sets_unpr -= {node}
            
            bjt.add_node(list(node_set_sets_unpr)[0])

        # Associate each ground rule with an appropriate BJT node that covers all and
        # only atoms occurring in the rule, then populate the BJT node with appropriate
        # input potential
        for atms, gr_rules in grounded_rules_by_atms.items():
            # Convert each rule to matching potential, and combine into single potential
            input_potentials = [
                _rule_to_potential(gr_rule, r_pr, atoms_map)
                for gr_rule, r_pr, _ in gr_rules
            ]

            if any(inp is None for inp in input_potentials):
                # Falsity included, program has no answer set
                return None

            if len(atms) > 0:
                inps_combined = _combine_factors_outer(input_potentials)
                bjt.nodes[atms]["input_potential"] = inps_combined
        
        # Null-potentials for BJT nodes without any input potentials
        for atms in bjt.nodes:
            if "input_potential" not in bjt.nodes[atms]:
                cases = {frozenset(case) for case in product(*[(ai, -ai) for ai in atms])}
                bjt.nodes[atms]["input_potential"] = {
                    case: { (frozenset(), frozenset()): Polynomial(float_val=1.0) }
                    for case in cases
                }

    return bjt


def _grounded_facts_only(rules):
    """ Test if set of rules consists only of grounded facts """
    return all(r.is_fact() and r.is_grounded() for r in rules)


def _ground_and_index(prog):
    """
    Ground program, construct a directed graph reflecting dependency between grounded
    atoms (by program rules), index grounded_rules by occurring atoms. Return the
    indexed grounded rules & mappings between grounded atoms and their integer indices.
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
    atoms_map_inv = {v: k for k, v in atoms_map.items()}

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
    #   2) Index the grounded rule by occurring atoms, positive or negative
    grounded_rules_by_atms = defaultdict(set)

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

            # Index and add this grounded rule with r_pr and index
            gr_rule = Rule(head=gr_rule.head, body=gr_body_filtered)
            occurring_atoms = frozenset([
                atoms_map[lit.as_atom()] for lit in gr_rule.head+gr_rule.body
            ])
            grounded_rules_by_atms[occurring_atoms].add((gr_rule, r_pr, ri))

    return grounded_rules_by_atms, atoms_map, atoms_map_inv


def _rule_to_potential(rule, r_pr, atoms_map):
    """
    Subroutine for converting a (grounded) rule weighted with probability r_pr into
    an appropriate input potential
    """
    # Edge case; body-less integrity constraint with absolute probability: falsity
    if len(rule.head+rule.body) == 0 and r_pr is None:
        return None

    # Use integer indices of the atoms
    atoms_by_ind = frozenset({atoms_map[lit.as_atom()] for lit in rule.head+rule.body})
    rh_by_ind = frozenset([
        atoms_map[hl] if hl.naf==False else -atoms_map[hl.as_atom()]
        for hl in rule.head
    ])
    rb_by_ind = frozenset([
        atoms_map[bl] if bl.naf==False else -atoms_map[bl.as_atom()]
        for bl in rule.body
    ])

    # We won't consider idiosyncratic cases with negative rule head literals
    assert all(hl>0 for hl in rh_by_ind)

    cases = {
        frozenset(case) for case in product(*[(ai, -ai) for ai in atoms_by_ind])
    }

    # Requirements of external support (for positive atoms), computed as follows:
    # 1) Any atoms positive in the case that are not the rule head...
    # 2) ... but head atom is exempt if body is true (i.e. the rule licenses head
    #    if body is true)
    pos_requirements = {
        case: frozenset({cl for cl in case if cl>0}) - \
            (rh_by_ind if rb_by_ind <= case else frozenset())
        for case in cases
    }

    # In fact, what we need for combination for each case is the **complement**
    # of positive atom requirements w.r.t. the domain; i.e. set of atoms whose
    # requirements for external support is cleared (either not present from the
    # beginning or requirement satisfied during message passing)
    pos_clearances = { 
        # (Clearance of positive requirements, full domain of atoms as reference)
        case: (atoms_by_ind - pos_requirements[case], atoms_by_ind)
        for case in cases
    }

    if r_pr is not None:
        r_pr_logit = logit(r_pr[0], large="a")
        potential = {
            # Singleton dict (with pos_non_req as key) as value
            frozenset(case): {
                # Rule weight of exp(w) missed in case of deductive violations (i.e.
                # when body is true but head is not)
                pos_clearances[case]:  Polynomial(float_val=1.0)
                    if (rb_by_ind <= case) and (len(rh_by_ind)==0 or not rh_by_ind <= case)
                    else Polynomial.from_primitive(r_pr_logit)
            }
            for case in cases
        }
    else:
        # r_pr of None signals 'absolute' rule (i.e. even stronger than hard-weighted
        # rule) that eliminates possibility of deductive violation (body true, head
        # false), yet doesn't affect model weights
        potential = {
            frozenset(case): { pos_clearances[case]: Polynomial(float_val=1.0) }
            for case in cases
            # Zero potential for deductive violations
            if not ((rb_by_ind <= case) and (len(rh_by_ind)==0 or not rh_by_ind <= case))
        }

    return potential


def _belief_propagation(bjt, node_set):
    """
    (Modified) Shafer-Shenoy belief propagation on binary join trees, whose input
    potential storage registers are properly filled in. Populate output belief storage
    registers as demanded by node_set.
    """
    # Corresponding node in BJT
    bjt_node = bjt.nodes[node_set]

    # Fetch input potentials for the BJT node
    input_potential = bjt_node["input_potential"]

    # Fetch incoming messages for the BJT node, if any
    incoming_messages = []
    for from_node_set, _, msg in bjt.in_edges(node_set, data="message"):
        if msg is None:
            # If message not computed already, compute it (once and for all)
            # and store it in the directed edge's storage register
            _compute_message(bjt, from_node_set, node_set)
            msg = bjt.edges[(from_node_set, node_set)]["message"]

        incoming_messages.append(msg)

    # Combine incoming messages; combine entries by multiplying 'fully-alive'
    # weights and 'half-alive' weights respectively
    if len(incoming_messages) > 0:
        msgs_combined = _combine_factors_outer(incoming_messages)

        # Final binary combination of (combined) input potentials & (combined)
        # incoming messages
        inps_msgs_combined = _combine_factors_outer(
            [input_potential, msgs_combined]
        )
    else:
        # Empty messages; just consider input potential
        inps_msgs_combined = input_potential

    # (Partial) marginalization down to domain for the node set for this BJT node
    output_beliefs = _marginalize_outer(inps_msgs_combined, node_set)

    # Weed out any subcases that still have lingering positive atom requirements
    # (i.e. non-complete positive clearances), then fully marginalize per case
    output_beliefs = {
        case: sum(
            [
                val for subcase, val in inner.items()
                if len(subcase[0])==len(subcase[1])
            ],
            Polynomial(float_val=0.0)
        )
        for case, inner in output_beliefs.items()
    }

    # Populate the output belief storage register for the BJT node
    bjt.nodes[node_set]["output_beliefs"] = output_beliefs


def _compute_message(bjt, from_node_set, to_node_set):
    """
    Recursive subroutine called during belief propagation for computing outgoing
    message from one BJT node to another; populates the corresponding directed edge's
    message storage register
    """
    # Fetch input potentials for the BJT node
    input_potential = bjt.nodes[from_node_set]["input_potential"]

    # Fetch incoming messages for the BJT node from the neighbors, except the
    # message recipient
    incoming_messages = []
    for neighbor_node_set, _, msg in bjt.in_edges(from_node_set, data="message"):
        if neighbor_node_set == to_node_set: continue    # Disregard to_node_set

        if msg is None:
            # If message not computed already, compute it (once and for all)
            # and store it in the directed edge's storage register
            _compute_message(bjt, neighbor_node_set, from_node_set)
            msg = bjt.edges[(neighbor_node_set, from_node_set)]["message"]

        incoming_messages.append(msg)
    
    # Combine incoming messages; combine entries by multiplying 'fully-alive'
    # weights and 'half-alive' weights respectively
    if len(incoming_messages) > 0:
        msgs_combined = _combine_factors_outer(incoming_messages)

        # Final binary combination of (combined) input potentials & (combined)
        # incoming messages
        inps_msgs_combined = _combine_factors_outer(
            [input_potential, msgs_combined]
        )
    else:
        inps_msgs_combined = input_potential

    # (Partial) Marginalization down to domain for the intersection of from_node_set
    # and to_node_set
    common_nodes = from_node_set & to_node_set
    outgoing_msg = _marginalize_outer(inps_msgs_combined, common_nodes)

    # Dismissable nodes; if some nodes occur in the message sender but not in the
    # message recipient, such nodes will never appear again at any point of the
    # onward message path (due to running intersection property of join trees).
    # The implication, that we exploit here for computational efficiency, is that
    # we can now stop caring about clearances of such dismissable nodes, and if
    # needed, safely eliminating subcases that still have lingering uncleared
    # requirements of dismissable nodes.
    dismissables = from_node_set - to_node_set
    if len(dismissables) > 0:
        outgoing_msg = {
            case: {
                (subcase[0]-dismissables, subcase[1]-dismissables): val
                for subcase, val in inner.items()
                if len(dismissables & (subcase[1]-subcase[0])) == 0
            }
            for case, inner in outgoing_msg.items()
        }

    # Populate the outgoing message storage register for the BJT edge
    bjt.edges[(from_node_set, to_node_set)]["message"] = outgoing_msg


def _combine_factors_outer(factors):
    """
    Subroutine for combining a set of input factors at the outer-layer; entries
    are combined by calling _combine_factors_inner()
    """
    assert len(factors) > 0
    # Each factors are specifications of cases sharing the same atom set signature,
    # differing only whether elements are positive or negative
    assert all(
        len({frozenset([abs(a) for a in ff]) for ff in f})==1
        for f in factors
    )

    # Efficient factor combination by exploiting factorization with common atom
    # signatures and respective complements

    # Find atoms common to all signatures of the factors, then partition each
    # case in the factors according to the partition
    factor_signatures = [list(f)[0] for f in factors]
    signature_common = frozenset.intersection(*factor_signatures)
    signature_common_pn = frozenset(
        sum([[atm, -atm] for atm in signature_common], [])
    )
    signature_diffs_pn = [
        frozenset(
            sum([[atm, -atm] for atm in f_sig-signature_common], [])
        )
        for f_sig in factor_signatures
    ]
    factors_partitioned = [
        {(ff&signature_common_pn, ff&sig_diff) for ff in f}
        for f, sig_diff in zip(factors, signature_diffs_pn)
    ]

    # Collect factor-specific cases by common cases
    factored_cases = defaultdict(lambda: defaultdict(set))
    for fi, per_factor in enumerate(factors_partitioned):
        for f_common, f_uniq in per_factor:
            factored_cases[fi][f_common].add(f_uniq)

    # Combine and expand each product of factor-specific cases
    valid_cases_common = set.intersection(*[
        set(cases_common) for cases_common in factored_cases.values()
    ])
    valid_cases_by_common = defaultdict(list)
    for case_common in valid_cases_common:
        for fi, f_common in factored_cases.items():
            valid_cases_by_common[case_common].append(
                frozenset(f_common[case_common])
            )
    valid_cases = {
        case_common: [
            case_sp for case_sp in product(*case_specifics)
            if _literal_set_is_consistent(frozenset.union(*case_sp))
        ]
        for case_common, case_specifics in valid_cases_by_common.items()
    }

    # Compute entry values for possible cases considered
    combined_factor = {}
    for case_common, case_specifics in valid_cases.items():
        for case_sp in case_specifics:
            # Corresponding entry fetched from the factor
            entries_per_factor = [
                factors[i][frozenset.union(c, case_common)]
                for i, c in enumerate(case_sp)
            ]

            # Ensure combination is happening at the outer layer for all factors
            assert isinstance(entries_per_factor[0], dict)
            assert all(
                type(e) == type(entries_per_factor[0]) for e in entries_per_factor
            )

            case_union = frozenset.union(case_common, *case_sp)

            # Outer-layer combination where entries can be considered 'mini-factors'
            # defined per postive atom clearances
            combined_factor[case_union] = _combine_factors_inner(entries_per_factor)

    return combined_factor


def _combine_factors_inner(factors):
    """
    Subroutine for combining a set of input factors at the inner-layer; entries
    (which are Polynomials) are multiplied then marginalized by case union
    """
    assert len(factors) > 0

    # Compute entry values for possible cases considered
    combined_factor = defaultdict(lambda: Polynomial(float_val=0.0))
    for case in product(*factors):
        # Union of case specification
        case_union = (
            frozenset.union(*[c[0] for c in case]),
            frozenset.union(*[c[1] for c in case]),
        )

        # Corresponding entry fetched from the factor
        entries_per_factor = [factors[i][c] for i, c in enumerate(case)]

        # Ensure combination is happening at the outer layer for all factors
        assert isinstance(entries_per_factor[0], Polynomial)
        assert all(type(e) == type(entries_per_factor[0]) for e in entries_per_factor)

        combined_factor[case_union] += reduce(operator.mul, entries_per_factor)

    return dict(combined_factor)


def _combine_factors_simple(factors):
    """
    Subroutine for combining a set of 'simple' factors without layers (likely those
    stored in 'output_beliefs' register for a BJT node); entries are combined by
    multiplication
    """
    assert len(factors) > 0

    # Compute entry values for possible cases considered
    combined_factor = {}
    for case in product(*factors):
        # Union of case specification
        case_union = frozenset.union(*case)

        # Incompatible, cannot combine
        if not _literal_set_is_consistent(case_union): continue

        # Corresponding entry fetched from the factor
        entries_per_factor = [factors[i][c] for i, c in enumerate(case)]

        # Ensure combination is happening with polynomial values
        assert isinstance(entries_per_factor[0], Polynomial)
        assert all(type(e) == type(entries_per_factor[0]) for e in entries_per_factor)

        # Outer-layer combination where entries can be considered 'mini-factors'
        # defined per postive atom clearances
        combined_factor[case_union] = reduce(operator.mul, entries_per_factor)

    return combined_factor


def _marginalize_outer(factor, node_set):
    """
    Subroutine for (partially) marginalizing a factor at the outer-layer (while
    maintaining subdivision by positive requirement clearance) down to some domain
    specified by node_set
    """
    marginalized_factor = {}

    for case in product(*[(ai, -ai) for ai in node_set]):
        case = frozenset(case)

        outer_marginals = defaultdict(lambda: Polynomial(float_val=0.0))
        for f_case, f_inner in factor.items():
            if not f_case >= case: continue     # Irrelevant f_case

            for pos_clrs, val in f_inner.items():
                outer_marginals[pos_clrs] += val

        marginalized_factor[case] = dict(outer_marginals)

    return marginalized_factor


def _marginalize_simple(factor, node_set):
    """
    Subroutine for simple marginalization of factor without layers (likely those
    stored in 'output_beliefs' register for a BJT node) down to some domain specified
    by node_set
    """
    marginalized_factor = defaultdict(lambda: Polynomial(float_val=0.0))

    for case in product(*[(ai, -ai) for ai in node_set]):
        case = frozenset(case)

        for f_case, f_val in factor.items():
            if not f_case >= case: continue     # Irrelevant f_case
            marginalized_factor[case] += f_val

    return dict(marginalized_factor)


def _literal_set_is_consistent(lit_set):
    """
    Subroutine for checking if a set of literals (represented with signed integer
    indices) is consistent; i.e. doesn't contain a literal and its negation at the
    same time
    """
    atm_set = {abs(lit) for lit in lit_set}

    # Inconsistent if and only if lit_set contained both atm & -atm for some atm,
    # which would be reduced to atm in atm_set
    return len(atm_set) == len(lit_set)
