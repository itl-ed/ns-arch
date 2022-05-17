import numpy as np


class Exemplars:
    """
    Inventory of exemplars encountered, stored as feature vectors (output from vision
    module's feature extractor component). Primarily used for incremental few-shot
    registration of novel concepts.
    """
    def __init__(self):
        # Dict from visual concept to list of vectors (matrix)
        self.entries = {}

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"Exemplars(len={len(self)})"
    
    def __getitem__(self, concept):
        return self.entries.get(concept)

    def add(self, concept, f_vec):
        if concept in self.entries:
            self.entries[concept] = np.concatenate((self.entries[concept], f_vec[None]))
        else:
            self.entries[concept] = f_vec[None]
