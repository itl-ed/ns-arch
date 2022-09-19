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
from collections import defaultdict

import tqdm
import numpy as np
import matplotlib.pyplot as plt

from itl.opts import parse_arguments


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
            ax.text(i, j, round(data[j, i].item(), 1), ha="center", va="center")


if __name__ == "__main__":
    opts = parse_arguments()

    res_dir = os.path.join(opts.output_dir_path, "exp1_res")
    results_confMat = defaultdict(dict)
    results_curve = defaultdict(dict)

    # Collect result
    dir_contents = list(os.listdir(res_dir))
    for data in tqdm.tqdm(dir_contents, total=len(dir_contents)):
        name_parse = re.findall(r"(.+)_(.+)_(.+)_(.+)_(\d+)", data)[0]
        data_type, *exp_config, seed = name_parse
        diff, feedStratT, initStratL = exp_config
        
        if data_type == "confMat":
            # Confusion matrix data
            with open(os.path.join(res_dir, data)) as data_f:
                reader = csv.reader(data_f)
                test_set_size = int(next(reader)[0])
                concepts = next(reader)

                confMat = np.array([[float(d) for d in row] for row in reader])
                confMat = confMat * test_set_size

                if (feedStratT, initStratL) in results_confMat[diff]:
                    stats_agg = results_confMat[diff][(feedStratT, initStratL)]
                    stats_agg["confMat"] += confMat
                    stats_agg["test_set_size"] += test_set_size
                    assert stats_agg["concepts"] == concepts
                else:
                    results_confMat[diff][(feedStratT, initStratL)] = {
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

                if (feedStratT, initStratL) in results_curve[diff]:
                    stats_agg = results_curve[diff][(feedStratT, initStratL)]
                    stats_agg["curve"] += curve
                    stats_agg["trials"] += 1
                else:
                    results_curve[diff][(feedStratT, initStratL)] = {
                        "curve": curve,
                        "trials": 1
                    }

        else:
            continue

    # Pre-defined ordering for listing legends
    config_ord = [
        "zeroInit_semOnly_minHelp", "zeroInit_semOnly_medHelp", "zeroInit_semOnly_maxHelp",
        "zeroInit_semWithImpl_minHelp", "zeroInit_semWithImpl_medHelp", "zeroInit_semWithImpl_maxHelp",
        "mixInitV_semOnly_minHelp", "mixInitV_semOnly_medHelp", "mixInitV_semOnly_maxHelp",
        "mixInitV_semWithImpl_minHelp", "mixInitV_semWithImpl_medHelp", "mixInitV_semWithImpl_maxHelp",
        "mixInitVL_semOnly_minHelp", "mixInitVL_semOnly_medHelp", "mixInitVL_semOnly_maxHelp",
        "mixInitVL_semWithImpl_minHelp", "mixInitVL_semWithImpl_medHelp", "mixInitVL_semWithImpl_maxHelp",
    ]

    # Aggregate and visualize: curve
    for diff, agg_stats in results_curve.items():
        fig = plt.figure()

        for exp_config, data in agg_stats.items():
            feedStratT, initStratL = exp_config
            ## Temp: rewriting config names
            feedStratT = feedStratT + "Help"
            if diff == "base":
                diff = "nonFine"
            elif diff == "easy":
                diff = "fineEasy"
            elif diff == "hard":
                diff = "fineHard"

            plt.plot(
                range(1, len(data["curve"])+1),
                data["curve"] / data["trials"],
                label=f"zeroInit_semOnly_{feedStratT}"
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
        
        plt.title(f"Learning curve for {diff} difficulty (N=50)")
        plt.savefig(os.path.join(opts.output_dir_path, f"curve_{diff}.png"))

    # Aggregate and visualize: confusion matrix
    for diff, agg_stats in results_confMat.items():
        for exp_config, data in agg_stats.items():
            feedStratT, initStratL = exp_config
            ## Temp: rewriting config names
            feedStratT = feedStratT + "Help"
            if diff == "base":
                diff = "nonFine"
            elif diff == "easy":
                diff = "fineEasy"
            elif diff == "hard":
                diff = "fineHard"

            config_label = f"zeroInit_semOnly_{feedStratT}"

            # Draw confusion matrix
            fig = plt.figure()
            draw_matrix(
                data["confMat"] / data["test_set_size"],
                data["concepts"], data["concepts"], fig.gca()
            )
            plt.suptitle(f"Confusion matrix for {diff} difficulty (N=50)", fontsize=16)
            plt.title(f"{config_label} agent", pad=18)
            plt.tight_layout()
            plt.savefig(os.path.join(opts.output_dir_path, f"confMat_{diff}_{config_label}.png"))

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

        print(f"Mean F1 scores ({diff}):")
        for d in data_collected:
            print("\t"+f"{d['config_label']}: {d['mF1']}")
        print("")
