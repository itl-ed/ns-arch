class Lexicon:
    """
    Lexicon of open-classed words <-> their denoted concepts in physical world. Should
    allow many-to-many mappings between symbols and denotations.
    """
    # Special reserved symbols
    RESERVED = {
        ("=", "*"): ("=", "*")      # 'Object identity' predicate
    }

    def __init__(self):
        self.s2d = {}  # Symbol-to-denotation
        self.d2s = {}  # Denotation-to-symbol

        # Add reserved symbols & denotations
        for s, d in Lexicon.RESERVED.items(): self.add(s, d)
    
    def __contains__(self, symbol):
        return symbol in self.s2d

    def add(self, symbol, denotation):
        # Symbol-to-denotation
        if symbol in self.s2d:
            self.s2d[symbol].append(denotation)
        else:
            self.s2d[symbol] = [denotation]
        
        # Denotation-to-symbol
        if denotation in self.d2s:
            self.d2s[denotation].append(symbol)
        else:
            self.d2s[denotation] = [symbol]

    def fill_from_vision(self, predicates):
        """
        Populate empty lexicon with provided dicts of visual concept predicates
        """
        assert len(self.s2d) == len(Lexicon.RESERVED), \
            "This method initializes empty lexicons only"

        for i, c in enumerate(predicates["cls"]):
            self.add((c.split(".")[0], "n"), (i, "cls"))
        for i, a in enumerate(predicates["att"]):
            self.add((a.split(".")[0], "a"), (i, "att"))
        for i, r in enumerate(predicates["rel"]):
            self.add((r.split(".")[0], "v"), (i, "rel"))
