import numpy as np


class Exemplars:
    """
    Inventory of exemplars encountered, stored as feature vectors (output from vision
    module's feature extractor component). Primarily used for incremental few-shot
    registration of novel concepts.
    """
    def __init__(self):
        # Dict from visual concept to list of vectors (matrix)
        self.pos_exs = {}        # Positive exemplars
        self.neg_exs = {}        # Negative (but close?) exemplars

    def __len__(self):
        return len(self.pos_exs)

    def __repr__(self):
        return f"Exemplars(len={len(self)})"
    
    def __getitem__(self, item):
        return { "pos": self.pos_exs.get(item), "neg": self.neg_exs.get(item) }

    def add_pos(self, concept, f_vec, provenance):
        if concept in self.pos_exs:
            self.pos_exs[concept] = (
                np.concatenate((self.pos_exs[concept][0], f_vec[None])),
                np.concatenate((self.pos_exs[concept][1], provenance[None])),
            )
        else:
            self.pos_exs[concept] = (f_vec[None], provenance[None])

    def add_neg(self, concept, f_vec, provenance):
        if concept in self.neg_exs:
            self.neg_exs[concept] = (
                np.concatenate((self.neg_exs[concept][0], f_vec[None])),
                np.concatenate((self.neg_exs[concept][1], provenance[None])),
            )
        else:
            self.neg_exs[concept] = (f_vec[None], provenance[None])
