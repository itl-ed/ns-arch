class KnowledgeBase:
    """
    Knowledge base containing pieces of knowledge in the form of LP^MLN (weighted ASP)
    rules
    """
    def __init__(self):
        self.entries = []

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"KnowledgeBase(len={len(self)})"

    def add(self, rule, weight, source):
        self.entries.append((rule, weight, source))
