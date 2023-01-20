"""
Visual analysis of concept exemplar vectors in low-dimensional space
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import uuid
import random
import warnings
warnings.filterwarnings("ignore")

import umap
import umap.plot
import hydra
import tqdm as tqdm
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.svm import SVC
from bokeh.io import save
from bokeh.models import Title
from bokeh.layouts import column

from itl import ITLAgent


TAB = "\t"

OmegaConf.register_new_resolver(
    "randid", lambda: str(uuid.uuid4())[:6]
)
@hydra.main(config_path="../../itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Set up agent
    agent = ITLAgent(cfg)
    exemplars = agent.lt_mem.exemplars

    accs = {}
    for conc_type in ["cls", "att", "rel"]:
        if conc_type == "cls" or conc_type == "att":
            vectors = exemplars.storage_vec[conc_type]
            pos_exs_inds = exemplars.exemplars_pos[conc_type]
            neg_exs_inds = exemplars.exemplars_neg[conc_type]

            # Dimensionality reduction down to 2D by UMAP, for visual inspection
            mapper = umap.UMAP().fit(vectors)

            # Plot for each concept
            umap.plot.output_file(
                os.path.join(cfg.paths.outputs_dir, f"{conc_type}_embs.html")
            )

            plots = []
            for c in tqdm.tqdm(pos_exs_inds, total=len(pos_exs_inds), desc=f"{conc_type}_embs"):
                concept_name = agent.lt_mem.lexicon.d2s[(c, conc_type)][0][0]
                concept_name = concept_name.replace("/", "_")

                # Evaluating exemplar sets by binary classification performance on
                # random 90:10 train/test split
                pos_shuffled = random.sample(pos_exs_inds[c], len(pos_exs_inds[c]))
                pos_train = pos_shuffled[:int(0.9*len(pos_shuffled))]
                pos_test = pos_shuffled[int(0.9*len(pos_shuffled)):]
                neg_shuffled = random.sample(neg_exs_inds[c], len(neg_exs_inds[c]))
                neg_train = neg_shuffled[:int(0.9*len(neg_shuffled))]
                neg_test = neg_shuffled[int(0.9*len(neg_shuffled)):]

                X = vectors[pos_train + neg_train]
                y = ([1] * len(pos_train)) + ([0] * len(neg_train))

                # Fit classifier and run on test set
                bin_clf = SVC(C=1.0, gamma=0.1, probability=True, random_state=42)
                bin_clf.fit(X, y)
                bin_clf.predict_proba(vectors[pos_test])
                pos_results = bin_clf.predict_proba(vectors[pos_test])[:,1] > 0.5
                neg_results = bin_clf.predict_proba(vectors[neg_test])[:,0] > 0.5
                accs[(concept_name, conc_type)] = \
                    (pos_results.sum()+neg_results.sum()) / (len(pos_test)+len(neg_test))

                labels = [
                    "p" if fv_i in pos_exs_inds[c]
                        else "n" if fv_i in neg_exs_inds[c] else "."
                    for fv_i in range(len(vectors))
                ]
                hover_data = pd.DataFrame(
                    {
                        "index": np.arange(len(vectors)),
                        "label": labels
                    }
                )

                # Plot data and save
                p = umap.plot.interactive(
                    mapper, labels=labels, hover_data=hover_data,
                    color_key={ "p": "#3333FF", "n": "#CC0000", ".": "#A0A0A0" },
                    point_size=5
                )
                p.add_layout(Title(text=concept_name, align="center"), "above")
                plots.append(p)

            save(column(*plots))
    
    for (concept_name, conc_type), acc in accs.items():
        print(f"Accuracy for {concept_name} ({conc_type}): {acc:.3f}")

if __name__ == "__main__":
    main()
