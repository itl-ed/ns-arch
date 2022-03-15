import torch


class Lexicon:
    """
    Lexicon of open-classed words <-> their denoted concepts in physical world
    """
    def __init__(self, opts):
        self.s2d = {}  # Symbol-to-denotation
        self.d2s = {}  # Denotation-to-symbol

        ## Load lexicon from available resources if available, designated in opts
        # From vision module checkpoint
        if opts.load_checkpoint_path is not None:
            ckpt = torch.load(opts.load_checkpoint_path)
            if "predicates" in ckpt:
                preds = ckpt["predicates"]
                for i, c in enumerate(preds["cls"]):
                    self.add((c.split(".")[0], "n"), (i, "cls"))
                for i, a in enumerate(preds["att"]):
                    self.add((a.split(".")[0], "a"), (i, "att"))
                for i, r in enumerate(preds["rel"]):
                    self.add((r.split(".")[0], "p"), (i, "rel"))
    
    def __contains__(self, symbol):
        return symbol in self.s2d

    def add(self, symbol, denotation):
        self.s2d[symbol] = denotation
        self.d2s[denotation] = symbol
