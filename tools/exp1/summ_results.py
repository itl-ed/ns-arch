"""
Script for collecting and aggregating experiment result stats for final report
and visualization
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import re
import csv
import logging
from collections import defaultdict

import tqdm
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


## Plot matrix
def draw_matrix(data, row_labs, col_labs, ax):
    
    # Draw matrix
    ax.matshow(data, cmap="Wistia")

    # Show all ticks
    ax.set_xticks(np.arange(len(col_labs)), rotation=30)
    ax.set_yticks(np.arange(len(row_labs)))

    # Label with provided predicate names
    ax.set_xticklabels(col_labs)
    ax.set_yticklabels(row_labs)

    # Horizontal axis on top
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Annotate with data entries
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            ax.text(i, j, round(data[j, i].item(), 2), ha="center", va="center")


@hydra.main(config_path="../../itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    dirs_with_results = []
    outputs_root_dir = os.path.join(cfg.paths.root_dir, "outputs")
    for date_dir in os.listdir(outputs_root_dir):
        if not os.path.isdir(os.path.join(outputs_root_dir, date_dir)): continue

        for time_dir in os.listdir(os.path.join(outputs_root_dir, date_dir)):
            full_out_dir = os.path.join(outputs_root_dir, date_dir, time_dir)

            if not os.path.isdir(full_out_dir):
                continue
            
            if "exp1_res" in os.listdir(full_out_dir):
                dirs_with_results.append((date_dir, time_dir))

    results_confMat = defaultdict(dict)
    results_curve = defaultdict(dict)

    # Collect results
    for res_dir in dirs_with_results:
        res_dir = os.path.join(outputs_root_dir, *res_dir, "exp1_res")
        dir_contents = list(os.listdir(res_dir))

        for data in tqdm.tqdm(dir_contents, total=len(dir_contents)):
            name_parse = re.findall(r"(.+)_(.+)_(.+)_(.+)_(\d+)", data)[0]
            data_type, *exp_config, seed = name_parse
            diff, feedStratT, semStratL = exp_config
            
            if data_type == "confMat":
                # Confusion matrix data
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)
                    test_set_size = int(next(reader)[0])
                    concepts = next(reader)

                    confMat = np.array([[float(d) for d in row] for row in reader])
                    confMat = confMat * test_set_size

                    if (feedStratT, semStratL) in results_confMat[diff]:
                        stats_agg = results_confMat[diff][(feedStratT, semStratL)]
                        stats_agg["confMat"] += confMat
                        stats_agg["test_set_size"] += test_set_size
                        assert stats_agg["concepts"] == concepts
                    else:
                        results_confMat[diff][(feedStratT, semStratL)] = {
                            "confMat": confMat,
                            "test_set_size": test_set_size,
                            "concepts": concepts
                        }

            elif data_type == "curve":
                # Learning curve data
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)
                    _ = next(reader)

                    curve = np.array([int(row[1]) for row in reader])

                    if (feedStratT, semStratL) in results_curve[diff]:
                        stats_agg = results_curve[diff][(feedStratT, semStratL)]
                        stats_agg["curve"] += curve
                        stats_agg["trials"] += 1
                    else:
                        results_curve[diff][(feedStratT, semStratL)] = {
                            "curve": curve,
                            "trials": 1
                        }

            else:
                continue

    # Pre-defined ordering for listing legends
    config_ord = [
        "semOnly_minHelp", "semOnly_medHelp", "semOnly_maxHelp",
        "semWithImpl_minHelp", "semWithImpl_medHelp", "semWithImpl_maxHelp"
    ]

    # Aggregate and visualize: curve
    for diff, agg_stats in results_curve.items():
        fig = plt.figure()

        for exp_config, data in agg_stats.items():
            feedStratT, semStratL = exp_config
            ## Temp: rewriting config names
            if diff == "base":
                diff = "nonFine"
            elif diff == "easy":
                diff = "fineEasy"
            elif diff == "hard":
                diff = "fineHard"

            plt.plot(
                range(1, len(data["curve"])+1),
                data["curve"] / data["trials"],
                label=f"{semStratL}_{feedStratT}"
            )

        # Plot curve
        plt.xlabel("# exemplars")
        plt.ylabel("cumulative regret")
        plt.ylim(0, int(len(data["curve"]) * 0.8))
        plt.grid()

        # Ordering legends according to the prespecified ordering above
        handles, labels = plt.gca().get_legend_handles_labels()
        hls_sorted = sorted(
            [(h, l) for h, l in zip(handles, labels)],
            key=lambda x: config_ord.index(x[1])
        )
        handles = [hl[0] for hl in hls_sorted]
        labels = [hl[1] for hl in hls_sorted]
        plt.legend(handles, labels)
        
        plt.title(f"Learning curve for {diff} difficulty")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"curve_{diff}.png"))

    # Aggregate and visualize: confusion matrix
    for diff, agg_stats in results_confMat.items():
        for exp_config, data in agg_stats.items():
            feedStratT, semStratL = exp_config
            ## Temp: rewriting config names
            if diff == "base":
                diff = "nonFine"
            elif diff == "easy":
                diff = "fineEasy"
            elif diff == "hard":
                diff = "fineHard"

            config_label = f"semOnly_{feedStratT}"

            # Draw confusion matrix
            fig = plt.figure()
            draw_matrix(
                data["confMat"] / data["test_set_size"],
                data["concepts"], data["concepts"], fig.gca()        # Binary choice mode
                # data["concepts"][:-1], data["concepts"], fig.gca()     # Multiple choice mode
            )
            plt.suptitle(f"Confusion matrix for {diff} difficulty", fontsize=16)
            plt.title(f"{config_label} agent", pad=18)
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.paths.outputs_dir, f"confMat_{diff}_{config_label}.png"))

            # Compute aggregate metric: mean F1 score
            P_per_concept = [
                data["confMat"][i,i] / data["confMat"][:,i].sum()
                for i in range(len(data["confMat"]))
            ]
            R_per_concept = [
                data["confMat"][i,i] / data["test_set_size"]
                for i in range(len(data["confMat"]))
            ]
            f1_per_concept = [2*p*r/(p+r) for p, r in zip(P_per_concept, R_per_concept)]

            data["config_label"] = config_label
            data["mF1"] = sum(f1_per_concept) / len(data["concepts"])
    
    # Report mF1 scores on terminal
    print("")
    for diff, agg_stats in results_confMat.items():
        data_collected = [d for d in agg_stats.values()]
        data_collected = sorted(data_collected, key=lambda x: config_ord.index(x["config_label"]))

        ## Temp: rewriting config names
        if diff == "base":
            diff = "nonFine"
        elif diff == "easy":
            diff = "fineEasy"
        elif diff == "hard":
            diff = "fineHard"

        logger.info(f"Mean F1 scores ({diff}):")
        for d in data_collected:
            logger.info("\t"+f"{d['config_label']}: {d['mF1']}")
        print("")


if __name__ == "__main__":
    main()
