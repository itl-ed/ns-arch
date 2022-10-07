""" Program.split() factored out """
from collections import defaultdict

import heapq
import networkx as nx

from ..literal import Literal


def split_program(comp, atoms_map, grounded_rules_rel):
    """
    Search for a minimal splitting set for comp and return the corresponding split
    programs.

    (Implements paper [[How to Split a Logic Program]] (Ben-Eliyahu-Zohary, 2021).)
    """
    from .program import Program

    if len(comp.nodes) == 1 or len(grounded_rules_rel) == 1:
        # Don't even bother to split
        found_split = frozenset(comp)
    
    else:
        # Construct super-dependency graph for the provided component
        comp_sd = nx.DiGraph()

        # Node for each strongly connected component (SCC)
        sccs_map = {}; sccs_inv_map = {}
        for i, c in enumerate(nx.strongly_connected_components(comp)):
            sccs_map[i] = c
            for v in c: sccs_inv_map[v] = i
            comp_sd.add_node(i)

        # Edge for each edge connecting nodes in the original component
        for u, v in comp.edges:
            u_scc = {i for i, scc in sccs_map.items() if u in scc}
            v_scc = {i for i, scc in sccs_map.items() if v in scc}

            assert len(u_scc) == len(v_scc) == 1
            u_scc = u_scc.pop(); v_scc = v_scc.pop()

            # Add edge (only if between different SCC nodes)
            if u_scc != v_scc: comp_sd.add_edge(u_scc, v_scc)
        
        # For each node v in comp prepare values of tree(v) (the set of all nodes that
        # belongs to any SCC v_sd such that there is a path in comp_sd from v_sd to scc(v);
        # see original paper for more details)
        reachable_sccs = {
            v: nx.shortest_path(comp_sd, sccs_inv_map[v]).keys() for v in comp.nodes
        }
        nodes_in_reachable_sccs = {
            v: set.union(*[sccs_map[scc] for scc in sccs])
            for v, sccs in reachable_sccs.items()
        }
        trees = defaultdict(set)
        for v, nodes in nodes_in_reachable_sccs.items():
            for u in nodes:
                trees[u].add(v)
        
        # Find source nodes in the super-dependency graph, and find union of corresponding
        # nodes in the original comp; to be used as the initial state in search of minimal
        # splitting set
        sources = set.union(*[
            sccs_map[v_sd] for v_sd in comp_sd.nodes if comp_sd.in_degree[v_sd] == 0
        ])
        sources = frozenset(sources)

        # Perform uniform cost search until a minimal splitting set is found
        visited = set(); found_split = None
        pqueue = [(0, sources)]; heapq.heapify(pqueue)
        while len(pqueue) > 0:
            # Pop state with the lowest path cost
            current_path_cost, current_state = heapq.heappop(pqueue)

            # If no rules to add to bottom found, is in goal_state
            is_goal = True

            # Add to set of explored states
            visited.add(current_state)

            # Find the lowest rule (i.e. first find) that needs to be included in splitting
            # bottom; i.e. rule with a head in current state, and any of the other atoms
            # not in current state
            for r, _, _ in grounded_rules_rel:
                # Auxiliary heads of integrity constraints are never sources, and don't 
                # need to be included minimal splitting sets.
                if len(r.head) == 0:
                    continue

                head_overlaps = atoms_map[r.head[0].as_atom()] in current_state
                body_is_subset = {atoms_map[l.as_atom()] for l in r.body} <= current_state

                if head_overlaps and not body_is_subset:
                    # Not a goal state
                    is_goal = False

                    # Splitting bottom should be expanded; unite state with tree(r)
                    expanded_state = frozenset.union(
                        current_state,
                        *[trees[atoms_map[l.as_atom()]] for l in r.literals()]
                    )
                    expansion_cost = len(expanded_state) - len(current_state)
                    expanded_path_cost = current_path_cost + expansion_cost

                    if expanded_state in visited:
                        # Update path cost in pqueue if better
                        i = [s for _, s in pqueue].index(expanded_state)
                        pqueue[i][0] = expanded_path_cost   # Update cost (becomes non-heap)
                        heapq.heapify(pqueue)               # To valid heap again
                    else:
                        # Push to pqueue
                        heapq.heappush(pqueue, (expanded_path_cost, expanded_state))

                    # Finding one rule suffices
                    break
            
            if is_goal:
                # Minimal splitting set found; break
                found_split = current_state
        
        assert found_split is not None, "Must have found the trivial splitting set at least!"

    # Check whether the split is the trivial one with empty top
    is_trivial_split = len(found_split) == len(comp.nodes)

    # Obtain program bottom & top based on the found splitting set
    bottom_rules = []; top_rules = []

    # Add each grounded rule to either top or bottom
    for gr_rule, r_pr, ri in grounded_rules_rel:

        # Check if this grounded rule should enter bottom or top
        add_to_top = False
        if not is_trivial_split:
            literals = gr_rule.literals()
            if len(gr_rule.head) == 0:
                # Integrity constraint, account for auxiliary literal
                literals.add(Literal("con_aux", args=[(ri, False)]))

            for l in literals:
                if atoms_map[l.as_atom()] not in found_split:
                    add_to_top = True
                    break

        # Add a copy for each rule in self that unifies with the grounded rule, with
        # retrieved weight
        if add_to_top:
            top_rules.append((gr_rule, r_pr))
        else:
            bottom_rules.append((gr_rule, r_pr))

    # Return None as top for trivial splits
    bottom_program = Program(bottom_rules)
    top_program = Program(top_rules) if len(top_rules) > 0 else None

    return bottom_program, top_program, found_split
