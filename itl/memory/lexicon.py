class Lexicon:
    """
    Lexicon of open-classed words <-> their denoted concepts in physical world. Should
    allow many-to-many mappings between symbols and denotations.
    """
    # Special reserved symbols
    RESERVED = {
        ("=", "*"): ("=", "*"),     # 'Object identity' predicate
        ("?", "*"): ("?", "*")      # Wh-quantified predicate
    }

    def __init__(self):
        self.s2d = {}     # Symbol-to-denotation
        self.d2s = {}     # Denotation-to-symbol
        self.d_freq = {}  # Denotation frequency

        # Add reserved symbols & denotations
        for s, d in Lexicon.RESERVED.items(): self.add(s, d)
    
    def __contains__(self, symbol):
        return symbol in self.s2d

    def add(self, symbol, denotation, freq=None):
        # For consistency; we don't need the 'adjective satellite' thingy
        # from WordNet
        if symbol[1] == "s": symbol = (symbol[0], "a")

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

        freq = freq or 1
        self.d_freq[denotation] = freq

    def fill_from_dicts(self, predicates, predicates_freq):
        """
        Populate empty lexicon with provided dicts of visual concept predicates
        and their frequency info
        """
        assert len(self.s2d) == len(Lexicon.RESERVED), \
            "This method initializes empty lexicons only"

        for i, c in enumerate(predicates["cls"]):
            if predicates_freq is not None:
                freq = predicates_freq["cls"][i]
            else:
                freq = None
            lemma, pos, _ = c.split(".")
            self.add((lemma, pos), (i, "cls"), freq)

        for i, a in enumerate(predicates["att"]):
            if predicates_freq is not None:
                freq = predicates_freq["att"][i]
            else:
                freq = None
            lemma, pos, _ = a.split(".")
            self.add((lemma, pos), (i, "att"), freq)

        for i, r in enumerate(predicates["rel"]):
            if predicates_freq is not None:
                freq = predicates_freq["rel"][i]
            else:
                freq = None
            lemma, pos, _ = r.split(".")
            self.add((lemma, pos), (i, "rel"), freq)
