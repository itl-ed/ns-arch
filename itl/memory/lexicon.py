class Lexicon:
    """
    Lexicon of open-classed words <-> their denoted concepts in physical world
    """
    def __init__(self):
        self.s2d = {}  # Symbol-to-denotation
        self.d2s = {}  # Denotation-to-symbol
    
    def __contains__(self, symbol):
        return symbol in self.s2d

    def add(self, symbol, denotation):
        self.s2d[symbol] = denotation
        self.d2s[denotation] = symbol

    def fill_from_vision(self, predicates):
        """
        Populate empty lexicon with provided dicts of visual concept predicates
        """
        assert len(self.s2d) == 0, "This method initializes empty lexicons only"

        for i, c in enumerate(predicates["cls"]):
            self.add((c.split(".")[0], "n"), (i, "cls"))
        for i, a in enumerate(predicates["att"]):
            self.add((a.split(".")[0], "a"), (i, "att"))
        for i, r in enumerate(predicates["rel"]):
            self.add((r.split(".")[0], "p"), (i, "rel"))
