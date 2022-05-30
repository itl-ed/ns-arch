from ..lpmln import Program
from ..lpmln.utils import logit, sigmoid


class KnowledgeBase:
    """
    Knowledge base containing pieces of knowledge in the form of LP^MLN (weighted
    ASP) rules
    """
    def __init__(self):
        # Mapping from rule to set of provenances
        self.entries = {}

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"KnowledgeBase(len={len(self)})"

    def add(self, rule, weight, source):
        entry = self.entries.get(rule)
        if entry is None:
            self.entries[rule] = {(weight, source)}
        else:
            entry.add((weight, source))

    def export_as_program(self):
        kb_prog = Program()

        for rule, prov in self.entries.items():
            # For rules with multiple provenance, probabilities are aggregated by
            # summing in logit-space and then sigmoid-ing back to probability space
            w_pr = sigmoid(sum(logit(w) for w, _ in prov))
            kb_prog.add_rule(rule, w_pr)

        return kb_prog
