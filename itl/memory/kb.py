import re
from collections import defaultdict

from ..lpmln import Program
from ..lpmln.utils import logit, sigmoid


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

    def export_as_program(self):
        kb_prog = Program()

        ## TODO: casting into abductive program

        # for rule, prov in self.entries.items():
        #     # For rules with multiple provenance, probabilities are aggregated by
        #     # summing in logit-space and then sigmoid-ing back to probability space
        #     w_pr = sigmoid(sum(logit(w) for w, _ in prov))
        #     kb_prog.add_rule(rule, w_pr)

        return kb_prog
