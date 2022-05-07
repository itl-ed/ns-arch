"""
Implements LP^MLN program class
"""
import copy
import heapq
from collections import defaultdict
from itertools import product
from functools import reduce

import clingo
import numpy as np
import networkx as nx
from multiset import Multiset, FrozenMultiset

from .literal import Literal
from .rule import Rule
from .models import Models
from .topk_subset import topk_subset_gen


LARGE = 2e1           # Sufficiently large logit to use in place of, say, float('inf')
SCALE_PREC = 3e2      # For preserving some float weight precision
TOPK_RATIO = 0.75     # Percentage of answer sets to cover, by probability mass

class Program:
    """ Probabilistic ASP program, implemented as a list of weighted ASP rules. """
    def __init__(self, rules=None):
        self.rules = [] if rules is None else rules

        self._rules_by_atom = defaultdict(set)      
        for i, (rule, _) in enumerate(self.rules):
            for hl in rule.head:
                self._rules_by_atom[hl.as_atom()].add(i)
            for bl in rule.body:
                self._rules_by_atom[bl.as_atom()].add(i)

    def __len__(self):
        return len(self.rules)

    def __str__(self):
        """ LP^MLN program string representation """
        prog_s = ""

        weight_strs = []; max_ws_len = 0
        for _, w_pr in self.rules:
            if 1.0 in w_pr:
                weight_strs.append("a")
            elif 0.0 in w_pr:
                weight_strs.append("-a")
            else:
                if len(w_pr) == 1:
                    w_pr_str = f"logit({w_pr[0]:.3f})"
                else:
                    w_pr_str = ",".join([f"logit({p:.3f})" for p in w_pr])
                    w_pr_str = f"[{w_pr_str}]"
                weight_strs.append(w_pr_str)
            
            max_ws_len = max(len(weight_strs[-1]), max_ws_len)

        for (rule, _), ws in zip(self.rules, weight_strs):
            ws = ws + (" " * (max_ws_len-len(ws)))
            prog_s += f"{ws} ::   {str(rule)}\n"

        return prog_s

    def __repr__(self):
        return f"Program(len={len(self)})"

    def __add__(self, other):
        assert isinstance(other, Program), "Added value must be a Program"

        return Program(self.rules + other.rules)
    
    def __iadd__(self, other):
        return self + other
    
    def add_rule(self, rule, w_pr=None):
        """
        Probability value of 0 or 1 indicates hard-weighted rules. If w_pr is not provided,
        assume value(s) of 0.5 (effectively giving zero weights)
        """
        if len(rule.head) > 0:
            if w_pr is None:
                w_pr = [0.5] * len(rule.head)
            if type(w_pr) != list:
                w_pr = [w_pr] * len(rule.head)
        else:
            w_pr = [w_pr]
        w_pr = tuple(w_pr)

        assert isinstance(rule, Rule), "Added value must be a Rule"
        for p in w_pr:
            assert 0 <= p <= 1, "Must provide valid probability value to compute rule weight"

        self.rules.append((rule, w_pr))

        for hl in rule.head:
            self._rules_by_atom[hl.as_atom()].add(len(self.rules)-1)
        for bl in rule.body:
            self._rules_by_atom[bl.as_atom()].add(len(self.rules)-1)
    
    def add_hard_rule(self, rule):
        self.add_rule(rule, 1.0)

    def solve(self, topk_ratio=TOPK_RATIO, provided_mem=None):
        """
        Recursively find program solve results, and return as a set of independent decision
        trees. Each child of a node in a tree represents a possible answer set for the program,
        along with the probability value for the model. The forest can be used to answer queries
        on marginal probabilities of atoms. Some low-probability models may be pruned, resulting
        in the total joint probabilities not summing to one.

        Can exploit models obtained from previously solved idential (sub)programs, if provided as
        the argument `provided_mem`. Basically top-down dynamic programming.
        """
        provided_mem = {} if provided_mem is None else provided_mem

        # Can take shortcut if program consists only of grounded facts
        grounded_facts_only = all([
            r.is_fact() and r.is_grounded() for r, _ in self.rules
        ])
        if grounded_facts_only:
            facts = [(r.head[0], float(w_pr[0]), None) for r, w_pr in self.rules]

            models = Models(factors=facts)
            memoized_models = { FrozenMultiset(self.rules): models }

            return models, memoized_models

        # Feed compiled program string to clingo.Control object and ground program
        rules_obs = _Observer()
        ctl = clingo.Control()
        ctl.register_observer(rules_obs)
        ctl.add("base", [], self._pure_ASP_str())
        ctl.ground([("base", [])])

        atoms_map = {
            Literal.from_clingo_symbol(atom.symbol): atom.literal
            for atom in ctl.symbolic_atoms
        }
        atoms_inv_map = {v: k for k, v in atoms_map.items()}
        aux_i = len(atoms_map)

        # Index rules by grounded atoms. When tracking which original rules have yielded
        # the grounded rules below, relevant rules may be directly retrieved without having
        # to iterate over self.rules for every grounded rule.
        rules_by_atom_inv = defaultdict(set)
        for atm, ris in self._rules_by_atom.items():
            for ri in ris:
                rules_by_atom_inv[ri].add(atm)
        rules_by_atom_set = defaultdict(set)
        for ri, atms in rules_by_atom_inv.items():
            rules_by_atom_set[FrozenMultiset(atms)].add(ri)
        rules_by_atom_set = dict(rules_by_atom_set)

        instantiable_atoms = {
            ra: {
                ma for ma in atoms_map
                if ra.name == ma.name and len(ra.args) == len(ma.args) and all([
                    rarg[1] == True or rarg[0] == marg[0]
                    for rarg, marg in zip(ra.args, ma.args)
                ])
            }
            for ra in self._rules_by_atom
        }

        # For any atom sets including non-grounded atoms, fully instantiate
        for atms in rules_by_atom_set:
            grounded_atms = [{a} for a in atms if a.is_grounded()]
            non_grounded_atms = [
                instantiable_atoms[a] for a in atms if not a.is_grounded()
            ]

            if len(non_grounded_atms) > 0:
                for instance in product(*grounded_atms, *non_grounded_atms):
                    instance = FrozenMultiset(instance)
                    rules_by_atom_set[instance] = rules_by_atom_set[atms]

        # Iterate over the grounded rules for the following processes:
        #   1) Track which rule in the original program could have instantiated each
        #        grounded rule (wish clingo python API supported such feature...)
        #   2) Construct dependency graph from grounded rules
        dep_graph = nx.DiGraph()
        grounded_rules = FrozenMultiset()
        grounded_rules_by_head = defaultdict(FrozenMultiset)
        for head, body, choice in rules_obs.rules:
            # Find the original rule in the program that this grounded rule unifies with
            head_inv = [
                atoms_inv_map[hl] if hl > 0 else atoms_inv_map[abs(hl)].flip()
                for hl in head
            ]
            body_inv = [
                atoms_inv_map[bl] if bl > 0 else atoms_inv_map[abs(bl)].flip()
                for bl in body
            ]
            gr_rule = Rule(head=head_inv, body=body_inv)

            rule_copies = Multiset()
            occurring_atms = FrozenMultiset([l.as_atom() for l in gr_rule.literals()])
            for ri in rules_by_atom_set[occurring_atms]:
                rule, w_pr = self.rules[ri]

                orig_soft = not ((1.0 in w_pr) or (0.0 in w_pr))
                if orig_soft != choice:
                    # Soft-hard weight mismatch; continue
                    continue

                rule_copies.add((gr_rule, w_pr))

            grounded_rules = grounded_rules + rule_copies

            if len(head) > 0:
                grounded_rules_by_head[head[0]] = \
                    grounded_rules_by_head[head[0]] + rule_copies

                for h_lit in head:
                    dep_graph.add_node(abs(h_lit))
                    for b_lit in body:
                        dep_graph.add_node(abs(b_lit))
                        dep_graph.add_edge(abs(b_lit), abs(h_lit))
                    
            else:
                # Integrity constraint; add rule-specific auxiliary atom
                aux_i += 1
                grounded_rules_by_head[aux_i] = \
                    grounded_rules_by_head[aux_i] + rule_copies

                for b_lit in body:
                    dep_graph.add_node(abs(b_lit))
                    dep_graph.add_node(aux_i)
                    dep_graph.add_edge(abs(b_lit), aux_i)

        # Try exploiting memoized solutions for the whole grounded rule set
        if len(provided_mem) > 0:
            if grounded_rules in provided_mem:
                models = provided_mem[grounded_rules]
                return models, provided_mem

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
            sum([grounded_rules_by_head[a] for a in comp.nodes()], FrozenMultiset())
            for comp in comps
        ]

        # Try exploiting memoized solutions by assembling from provided_mem
        if len(provided_mem) > 0:
            retrieved = []
            for gr_c in grounded_rules_per_comp:
                if gr_c not in provided_mem: break
                retrieved.append(provided_mem[gr_c])
            
            if len(retrieved) == len(comps):
                # Successfully recovered
                models = Models(retrieved)
                memoized_models = {
                    sum(grounded_rules_per_comp, FrozenMultiset()): models,
                    **provided_mem
                }
                return models, memoized_models

        indep_trees = []; memoized_models = {}
        for ci, comp in enumerate(comps):
            print(f"A> Let me see... ({ci+1}/{len(comps)})", end="\r")

            # Find possibly relevant rules for each component, ignoring grounded rules that
            # do not overlap at all
            grounded_rules_relevant = grounded_rules_per_comp[ci]

            if len(provided_mem) > 0:
                # Check if memoized solution exists for the component
                gr_rules_fms = FrozenMultiset(grounded_rules_relevant)
                if gr_rules_fms in provided_mem:
                    # Model found
                    indep_trees.append(provided_mem[gr_rules_fms])
                    memoized_models[gr_rules_fms] = provided_mem[gr_rules_fms]
                    continue

            # Try splitting the program component
            bottom, top = self._split(comp, atoms_map, grounded_rules_relevant)
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
                        self._rules_by_atom[atoms_inv_map[v]] for v in comp.nodes
                        if atoms_inv_map[v] in self._rules_by_atom
                    ]

                    # (Assuming only one fact is present with the atom as head)
                    facts = [self.rules[fs.pop()] for fs in facts]
                    facts = [(f.head[0], float(w_pr[0]), None) for f, w_pr in facts]

                    tree = Models(factors=facts)

                else:
                    # Solve the bottom (== comp) with clingo and return answer sets (with
                    # associated probabilities)
                    ctl = clingo.Control()
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
                        for m in models
                    ]
                    models = [
                        (
                            [Literal.from_clingo_symbol(a) for a in atoms],
                            sum([a.arguments[1].number / SCALE_PREC for a in unsats])
                        )
                        for atoms, unsats in models
                    ]
                    logZ = reduce(np.logaddexp, [weight for _, weight in models])
                    models = [(atoms, np.exp(weight-logZ)) for atoms, weight in models]

                    outcomes = [(weight, atoms, None) for atoms, weight in models]
                    tree = Models(outcomes=outcomes)
                
                # Memoize bottom
                memoized_models[FrozenMultiset(bottom.rules)] = tree

            else:
                # Solve the bottom for possible models with probabilities, obtain & solve the
                # reduced top for each model
                if grounded_facts_only:
                    # If program only consists of grounded choice facts, may bypass clingo and
                    # find models along with probabilities combinatorially, possibly pruning
                    # low-probability models from the bottom.
                    # (Cannot use factored representation since we need to reduce program top for
                    # each possible model of the program bottom.)
                    
                    # Only need to consider soft rules (i.e. rules with 0.0 < w_pr < 1.0) when finding
                    # top-k models with this method
                    soft_facts = [(rule, w_pr) for rule, w_pr in bottom.rules if 0.0 < w_pr[0] < 1.0]
                    hard_facts = [
                        (rule, w_pr) for rule, w_pr in bottom.rules if w_pr[0] == 0.0 or w_pr[0] == 1.0
                    ]

                    # Aggregate rules with same head atom; probabilities are logsumexp-ed (then exp-ed back)
                    soft_facts_agg = defaultdict(lambda: float("-inf"))
                    for rule, w_pr in soft_facts:
                        soft_facts_agg[rule.head[0]] = np.exp(np.logaddexp(soft_facts_agg[rule.head[0]], np.log(w_pr)))
                    soft_facts = [(Rule(head=head), w_pr) for head, w_pr in soft_facts_agg.items()]

                    # Rules should be sorted by weights first to apply the algorithm
                    soft_facts = sorted(soft_facts, key=lambda rw: rw[1][0], reverse=True)

                    # Using logits of w_pr values as rule weights ensures direct association of the
                    # rule weights with the marginal probabilities of rule head atoms across all possible
                    # models (... on the assumption that there are no probabilistic choice rules with the
                    # same head atoms with non-disjoint body in program)
                    rule_weights = [_logit(w_pr[0]) for _, w_pr in soft_facts]

                    # (Log of) partition function for all the soft rules can be analytically computed as below
                    logZ = sum([np.log(1+np.exp(w)) for w in rule_weights])

                    # Log of total probability mass covered, from the top; need to query models until more than
                    # aggregate probability mass gets larger than top_k
                    log_pmass_covered = float("-inf")        # Represents limit(log(x)) as x -> +0

                    # Collect most probable possible worlds
                    subsets = []
                    subset_generator = topk_subset_gen(rule_weights)
                    for subset, weight_sum in subset_generator:
                        # Update pmass_covered with log-sum-exp
                        log_joint_p = weight_sum - logZ
                        log_pmass_covered = np.logaddexp(log_pmass_covered, log_joint_p)

                        # Append model retrieved from the indices in subset along with joint probability
                        subsets.append((subset, np.exp(log_joint_p)))
                        
                        # Break with sufficient coverage
                        if log_pmass_covered >= np.log(topk_ratio):
                            break
                    subset_generator.close()

                    # Translate each subset to explicit representation set of positive/negative
                    # atoms
                    bottom_models = [
                        (
                            {sr.head[0] for i, (sr, _) in enumerate(soft_facts) if i in ss},
                            {sr.head[0] for i, (sr, _) in enumerate(soft_facts) if i not in ss},
                            lp
                        ) for ss, lp in subsets
                    ]

                    # Combine the results with hard-weighted facts
                    bottom_models = [
                        (
                            pos_atoms | {hr.head[0] for (hr, w_pr) in hard_facts if w_pr==1.0},
                            neg_atoms | {hr.head[0] for (hr, w_pr) in hard_facts if w_pr==0.0},
                            lp
                        )
                        for pos_atoms, neg_atoms, lp in bottom_models
                    ]

                else:
                    # Solve the bottom (== comp) with clingo and return answer sets (with
                    # associated probabilities)
                    raise NotImplementedError

                # Solve reduced program top for each discovered model; first compute program
                # reduction by the common atoms
                atom_sets = [
                    {(pl, True) for pl in bm[0]} | {(nl, False) for nl in bm[1]}
                    for bm in bottom_models
                ]
                atom_commons = set.intersection(*atom_sets)
                reduced_common = top._reduce(
                    {atm for atm, pos in atom_commons if pos},
                    {atm for atm, pos in atom_commons if not pos}
                )

                # Now for each bottom model reduce the common reduction with the remainder of
                # the atoms, and solve the fully reduced
                outcomes = []
                atom_diffs = [a-atom_commons for a in atom_sets]
                for bi, ((pos_atoms, _, pr), atoms) in enumerate(zip(bottom_models, atom_diffs)):
                    print(f"A> Let me see... ({bi+1}/{len(bottom_models)})", end="\r")

                    pos_atoms_diff = {atm for atm, pos in atoms if pos}
                    neg_atoms_diff = {atm for atm, pos in atoms if not pos}

                    reduced_top = reduced_common._reduce(pos_atoms_diff, neg_atoms_diff)
                    top_models, top_memoized = reduced_top.solve(provided_mem={
                        **memoized_models, **provided_mem
                    })

                    # Memoize reduced top & merge returned memoized models
                    memoized_models.update(top_memoized)
                    memoized_models[FrozenMultiset(reduced_top.rules)] = top_models

                    outcomes.append((pr, pos_atoms, top_models))
                tree = Models(outcomes=outcomes)

            indep_trees.append(tree)
            print("A>" + (" "*50), end="\r")
        
        models = Models(factors=indep_trees)

        # Memoize whole
        memoized_models[grounded_rules] = models

        return models, memoized_models
    
    def optimize(self, statements):
        """
        Solve for the optimal model as specified by [statements] argument. Optimal solution
        is found by composing appropriate optimization statements in clingo, which is attached
        to string representation of the program, and solving the program with clingo.
        
        Each optimization statement designates how much weight each occurrence of relevant
        literals (counted at most once for tuple) should contribute to the total weight.
        Statements that come earlier in the list will have higher priority.
        """
        stm_asp_str = ""
        for p, stm in enumerate(statements[::-1]):
            # Each statement may consist of more than one weight formulas: that is, (literals,
            # weight, terms) tuples; weights will be summed in the priority level
            max_or_min, formulas = stm
            assert max_or_min == "maximize" or max_or_min == "minimize", \
                "Can only 'maximize' or 'minimize'"

            formulas_asp_str = [
                f"{weight}@{p}{','+','.join(terms) if len(terms)>0 else ''} : "
                    f"{','.join([str(l) for l in lits]) if len(lits)>0 else ''}"
                for lits, weight, terms in formulas
            ]

            stm_asp_str += f"#{max_or_min} {{ {'; '.join(formulas_asp_str)} }}.\n"

        # Optimize with clingo
        ctl = clingo.Control()
        ctl.add("base", [], self._pure_ASP_str()+stm_asp_str)
        ctl.ground([("base", [])])
        ctl.configuration.solve.models = 0
        ctl.configuration.solve.opt_mode = "opt"

        models = []; best_cost = [float("inf")] * len(statements)
        with ctl.solve(yield_=True) as solve_gen:
            for m in solve_gen:
                models.append((m.symbols(atoms=True), m.cost))
                if m.cost[::-1] < best_cost[::-1]: best_cost = m.cost
                if solve_gen.get().unsatisfiable: break
        
        models = [m[0] for m in models if m[1] == best_cost]
        models = [
            [Literal.from_clingo_symbol(a) for a in m] for m in models
        ]

        return models

    def _split(self, comp, atoms_map, grounded_rules_rel):
        """
        Search for a minimal splitting set for comp and return the corresponding split
        programs.

        (Implements paper [[How to Split a Logic Program]] (Ben-Eliyahu-Zohary, 2021).)
        """
        if len(comp.nodes) == 1:
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

                comp_sd.add_edge(u_scc, v_scc)
            
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

                # Find the lowest (in index) rule that needs to be included in splitting
                # bottom; i.e. rule with a head in current state, and any of the other atoms
                # not in current state
                for r, _ in self.rules:
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
        for gr_rule, w_pr in grounded_rules_rel:

            # Check if this grounded rule should enter bottom or top
            add_to_top = False
            if not is_trivial_split:
                for l in gr_rule.literals():
                    if atoms_map[l.as_atom()] not in found_split:
                        add_to_top = True
                        break

            # Add a copy for each rule in self that unifies with the grounded rule, with
            # retrieved weight
            if add_to_top:
                top_rules.append((gr_rule, w_pr))
            else:
                bottom_rules.append((gr_rule, w_pr))

        # Return None as top for trivial splits
        bottom_program = Program(bottom_rules)
        top_program = Program(top_rules) if len(top_rules) > 0 else None

        return bottom_program, top_program
        
    def _reduce(self, pos_atoms, neg_atoms):
        """ Return the program obtained by reducing self with given values of atoms """
        reduced_rules = copy.deepcopy(self.rules)
        rules_to_del = set()

        for pa in pos_atoms:
            for ri in self._rules_by_atom[pa.as_atom()]:
                rule = reduced_rules[ri][0]
                if rule.body_contains(pa.flip()):
                    # Exclude this rule
                    rules_to_del.add(ri)
                    continue

                # Remove positive appearance
                rule.body.remove(pa)

        for na in neg_atoms:
            for ri in self._rules_by_atom[na.as_atom()]:
                rule = reduced_rules[ri][0]
                if rule.body_contains(na):
                    # Exclude this rule
                    rules_to_del.add(ri)
                    continue

                # Remove negative appearance
                rule.body.remove(na.flip())

        reduced_rules = [
            r for i, r in enumerate(reduced_rules) if i not in rules_to_del
        ]

        # Filter possible empty rules, generated by reducing headless integrity
        # constraints
        reduced_rules = [
            r for r in reduced_rules if len(r[0].head) > 0 or len(r[0].body) > 0
        ]

        return Program(reduced_rules)

    def _pure_ASP_str(self, unsats=False):
        """
        Return string compilation into to pure ASP program string understandable by
        clingo.

        Feeding unsats=True adds 'unsat' atoms, which enables tracking of model weight
        sums as well as violated choice rules.
        
        Feeding unsats=False is primarily for the purpose of computing dependency
        graph of the grounded program, or for preparing vanilla ASP program (i.e. not
        a LP^MLN program) not related to any type of probabilistic inference.
        """
        as_str = ""

        for ri, (rule, w_pr) in enumerate(self.rules):
            if len(rule.head) <= 1:
                if w_pr[0] == 1.0:
                    # Add as-is
                    as_str += str(rule) + "\n"
                elif w_pr[0] == 0.0:
                    # Turn into corresponding integrity constraint and add
                    as_str += str(rule.flip()) + "\n"
                else:
                    if len(rule.head) == 1:
                        as_str += rule.str_as_choice() + "\n"

                        if unsats:
                            # Add rule with unsat atom which is derived whenever
                            # rule head is not true
                            weight = int(_logit(w_pr[0]) * SCALE_PREC)
                            unsat_args = [(ri, False), (weight, False)]
                            unsat_rule = Rule(
                                head=Literal("unsat", unsat_args),
                                body=[rule.head[0].flip()]
                            )
                            as_str += str(unsat_rule) + "\n"
                    else:
                        if unsats:
                            # Add a modified rule with the same body but unsat atom
                            # as head, which is satisfied whenever constraint body is
                            # true
                            weight = int(_logit(w_pr[0]) * SCALE_PREC)
                            unsat_args = [(ri, False), (weight, False)]
                            unsat_rule = Rule(
                                head=Literal("unsat", unsat_args),
                                body=rule.body
                            )
                            as_str += str(unsat_rule) + "\n"
                        else:
                            # Even when unsats=False, should add a modified rule with
                            # an auxiliary atom as head when body contains more than
                            # one literal, so that body literals are connected in the
                            # dependency graph
                            if len(rule.body) > 1:
                                aux_args = [(ri, False)]
                                aux_rule = Rule(
                                    head=Literal("unsat", aux_args),
                                    body=rule.body
                                )
                                as_str += str(aux_rule) + "\n"
            else:
                # Choice rules with multiple head literals
                as_str += rule.str_as_choice() + "\n"

                if unsats:
                    raise NotImplementedError

        return as_str


class _Observer:
    """For tracking added grounded rules"""
    def __init__(self):
        self.rules = []
    def rule(self, choice, head, body):
        self.rules.append((head, body, choice))


def _logit(p):
    """ Compute logit of the probability value p, capped by LARGE value (+/-) """
    if p == 1:
        return LARGE
    elif p == 0:
        return -LARGE
    else:
        return np.log(p/(1-p))
