class Lexicon:
    """
    Lexicon of open-classed words <-> their denoted concepts in physical world
    """

    def __init__(self):
        self.s2d = {}  # Symbol-to-denotation
        self.d2s = {}  # Denotation-to-symbol

    def add(self, symbol, denotation):
        self.s2d[symbol] = denotation
        self.d2s[denotation] = symbol

    def fill_initial(self, dataset):

        for t, entries in dataset.annotations["predicates"].items():

            # if t == "attributes":
            #     for e in entries:
            #         self["a"][e["name"].lower()] = (t, e["id"])
            if t == "categories":
                for e in entries:
                    e_id = dataset.cat_to_supercat[e["id"]]
                    self.add(("n", e["supercategory"].lower()), (t, e_id))
            else:
                for e in entries:
                    self.add(("n", e["name"].lower()), (t, e["id"]))