""" Models().query() factored out """
from itertools import product, chain, combinations
from collections import defaultdict

from ..rule import Rule
from ..polynomial import Polynomial


def query(models, q_vars, event, per_assignment=True, per_partition=False):
    """
    Query the tree structure to estimate the likelihood of each possible answer
    to the provided question, represented as tuple of entities (empty tuple for
    y/n questions). For each entity tuple that have some possible models satisfying
    the provided event specification, compute and return the aggregate probability
    mass of such models. Return the Models instances obtained after appropriate
    filtering as well.

    If q_vars is None we have a yes/no (polar) question, where having a non-empty
    tuple as q_vars indicates we have a wh-question.
    """
    if type(event) != frozenset:
        try:
            # Treat as set
            event = frozenset(event)
        except TypeError:
            # Accept single-rule event and wrap in a set
            assert isinstance(event, Rule)
            event = frozenset([event])

    if q_vars is None:
        # Empty tuple representing no wh-quantified variables to test. () in the
        # returned dicts may as well be interpreted as the "Yes" answer.
        q_vars = ()

        # Event is already grounded
        ev_instances = { (): list(event) }
    else:
        # Assign some arbitrary order among the variables
        assert type(q_vars) == tuple, "Provide q_vars as tuple"

        # Set of atoms that appear in models covered by this Models instance
        atoms_covered = models.atoms()     # Effectively the union of all models covered

        # Set of entities and predicates (along w/ arity info - values defined only for
        # predicate q_var) occurring in atoms_covered
        if len(atoms_covered) > 0:
            ents = set.union(*[{a[0] for a in atm.args} for atm in atoms_covered])
        else:
            ents = set()

        preds = set((atm.name, len(atm.args)) for atm in atoms_covered)
        # For now, let's limit our answer to "what is X" questions to nouns: i.e. object
        # class categories...
        preds = {p for p in preds if p[0].startswith("cls")}

        pred_var_arities = {
            l for l in set.union(*[ev_rule.literals() for ev_rule in event])
            if l.name=="*_?"
        }
        pred_var_arities = {l.args[0][0]: len(l.args)-1 for l in pred_var_arities}

        # All possible grounded instances of event
        subs_options = product(*[
            [p[0] for p in preds if pred_var_arities[qv]==p[1]] if is_pred else ents
            for qv, is_pred in q_vars
        ])
        ev_instances = {
            s_opt: [
                ev_rule.substitute(
                    preds={
                        qv[0]: o for qv, o in zip(q_vars, s_opt)
                        if qv[0].startswith("P")
                    },
                    terms={
                        (qv[0], True): (o, False) for qv, o in zip(q_vars, s_opt)
                        if not qv[0].startswith("P")
                    }
                )
                for ev_rule in event
            ]
            for s_opt in subs_options
        }

        # Initial pruning of q_vars assignments that are not worth considering; may
        # disregard assignments yielding any body-less rules (i.e. facts) whose head
        # atom(s) does not appear in atoms_covered
        ev_instances = {
            assig: ev_ins for assig, ev_ins in ev_instances.items()
            if not any(
                len(r.body)==0 and not any(h in atoms_covered for h in r.head)
                for r in ev_ins
            )
        }

    # For normalizing covered probabilitiy masses
    models_pmass = models.compute_Z()

    ## This method can provide answers to the question in two ways:
    #   1) Filtered models per assignment: Filter models so that we have model sets
    #       for each valid assignment to q_vars
    #   2) Filtered models per answer: Filter models so that we have models sets for
    #       each 'strongly exhaustive answer' to the question, i.e. sets of valid
    #       assignments to q_vars (Groenendijk & Stokhof's question semantics).
    
    if per_assignment:
        # 1) Per-assignment filtering
        filtered_models = defaultdict(lambda: [models])

        for ri in range(len(event)):
            for assig, ev_ins in ev_instances.items():
                # Instantiated rule to use as model filter
                ev_rule = ev_ins[ri]

                filtered = filtered_models[assig]

                # Models satisfying rule head & body
                filtered_hb = filtered
                if len(ev_rule.head) > 0:
                    for lits in [ev_rule.head] + ev_rule.body:
                        lits = frozenset(lits)
                        filtered_hb = [f.filter(lits) for f in filtered_hb]
                else:
                    filtered_hb = []

                # Models not satisfying body
                filtered_nb = filtered
                if len(ev_rule.body) > 0:
                    # Negation of conjunction of body literals == Disjunction of
                    # negated body literals
                    body_neg = frozenset([bl.flip() for bl in ev_rule.body])
                    filtered_nb = [f.filter(body_neg) for f in filtered_nb]
                else:
                    filtered_nb = []

                filtered = filtered_hb + filtered_nb

                filtered_models[assig] = filtered

        per_assig = dict(filtered_models)
        per_assig = {
            assig: (
                fms,
                sum([Polynomial.ratio_at_limit(m.compute_Z(), models_pmass) for m in fms])
                    if models_pmass != Polynomial(float_val=0.0) else 0.0
            )
            for assig, fms in per_assig.items()
        }
    else:
        per_assig = None

    if per_partition:
        # 2) Per-answer filtering

        raise NotImplementedError       # Need update to comply with recent changes
        possible_answers = list(chain.from_iterable(
            combinations(ev_instances, l) for l in range(1,len(ev_instances)+1)
        ))

        filtered_models = defaultdict(lambda: [models])

        # Rule-by-rule basis filtering to actually obtain models satisfying the answer
        # associated with each partition
        for ri in range(len(event)):
            for ans in possible_answers:
                # Grounded rules to satisfy and violate, which define this partition
                rules_sat = [
                    ev_ins[ri] for assig, ev_ins in ev_instances.items()
                    if assig in ans
                ]
                rules_viol = [
                    ev_ins[ri] for assig, ev_ins in ev_instances.items()
                    if assig not in ans
                ]

                filtered = filtered_models[ans]

                # For each rule to satisfy, either both head and body should hold, or
                # body should not hold
                for sr in rules_sat:
                    # Models satisfying rule head & body
                    filtered_hb = filtered
                    if len(sr.head) > 0:
                        for lits in [sr.head] + sr.body:
                            filtered_hb = [f.filter(lits) for f in filtered_hb]
                    else:
                        filtered_hb = []

                    # Models not satisfying body
                    filtered_nb = filtered
                    if len(sr.body) > 0:
                        # Negation of conjunction of body literals == Disjunction of
                        # negated body literals
                        body_neg = [bl.flip() for bl in sr.body]
                        filtered_nb = [f.filter(body_neg) for f in filtered_nb]
                    else:
                        filtered_nb = []

                    filtered = filtered_hb + filtered_nb

                # For each rule to violate, body should hold whereas head should not
                # hold
                for vr in rules_viol:
                    if len(vr.body) > 0:
                        for bl in vr.body:
                            filtered = [f.filter(bl) for f in filtered]

                    if len(vr.head) > 0:
                        # Negation of disjunction of head literals == Conjunction of
                        # negated head literals
                        for hl in vr.head:
                            filtered = [f.filter(hl.flip()) for f in filtered]

                filtered_models[ans] = filtered

        per_exh_answer = dict(filtered_models)
        per_exh_answer = {
            ans: (
                models,
                sum([m.compute_marginals()[1] for m in models]) / models_pmass
                    if models_pmass > 0 else 0.0
            )
            for ans, models in per_exh_answer.items()
        }
    else:
        per_exh_answer = None

    return per_assig, per_exh_answer
