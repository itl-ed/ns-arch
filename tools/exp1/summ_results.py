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

    results_cumulRegs = defaultdict(dict)
    results_confMat = defaultdict(lambda: defaultdict(dict))
    results_learningCurve = defaultdict(dict)

    # Collect results
    for res_dir in tqdm.tqdm(dirs_with_results, total=len(dirs_with_results)):
        res_dir = os.path.join(outputs_root_dir, *res_dir, "exp1_res")
        dir_contents = list(os.listdir(res_dir))

        for data in dir_contents:
            name_parse = re.findall(r"(.+)_(.+)_(.+)_(.+)_(\d+)", data)[0]
            data_type, *exp_config, seed = name_parse
            diff, feedStratT, semStratL = exp_config

            if data_type == "cumulRegs":
                # Cumulative regret curve data
                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)
                    _ = next(reader)

                    curve = np.array([int(row[1]) for row in reader])

                    if (feedStratT, semStratL) in results_cumulRegs[diff]:
                        stats_agg = results_cumulRegs[diff][(feedStratT, semStratL)]
                        for i, rg in enumerate(curve):
                            if i in stats_agg:
                                stats_agg[i].append(int(rg))
                            else:
                                stats_agg[i] = [int(rg)]
                    else:
                        results_cumulRegs[diff][(feedStratT, semStratL)] = {
                            i: [int(rg)] for i, rg in enumerate(curve)
                        }

            elif data_type.startswith("confMat"):
                # Confusion matrix data
                num_exs = int(data_type.strip("confMat"))

                with open(os.path.join(res_dir, data)) as data_f:
                    reader = csv.reader(data_f)
                    concepts = next(reader)

                    confMat = np.array([[float(d) for d in row] for row in reader])

                    if (feedStratT, semStratL) in results_confMat[num_exs][diff]:
                        stats_agg = results_confMat[num_exs][diff][(feedStratT, semStratL)]
                        stats_agg["matrix"].append(confMat)
                        stats_agg["num_test_suites"] += 1
                        assert stats_agg["concepts"] == concepts
                    else:
                        results_confMat[num_exs][diff][(feedStratT, semStratL)] = {
                            "matrix": [confMat],
                            "num_test_suites": 1,
                            "concepts": concepts
                        }

            else:
                continue

    # Pre-defined ordering for listing legends
    config_ord = [
        "semOnly_minHelp", "semOnly_medHelp", "semOnly_maxHelp",
        "semNeg_minHelp", "semNeg_medHelp", "semNeg_maxHelp",
        "semNegScal_minHelp", "semNegScal_medHelp", "semNegScal_maxHelp"
        # Among these, semNeg/semNegScal_min/medHelp are vacuous config combinations
    ]

    # Aggregate and visualize: cumulative regret curve
    for diff, agg_stats in results_cumulRegs.items():
        fig, ax = plt.subplots()

        for exp_config, data in agg_stats.items():
            feedStratT, semStratL = exp_config
            stats = [
                (i, np.mean(rgs), 1.96 * np.std(rgs)/np.sqrt(len(rgs)))
                for i, rgs in data.items()
            ]

            # Plot mean curve
            curve = ax.plot(
                [i+1 for i, _, _ in stats],
                [mrg for _, mrg, _ in stats],
                label=f"{semStratL}_{feedStratT}"
            )
            # Plot confidence intervals
            ax.fill_between(
                [i+1 for i, _, _ in stats],
                [mrg-cl for _, mrg, cl in stats],
                [mrg+cl for _, mrg, cl in stats],
                color=curve[0].get_color(), alpha=0.2
            )

        # Plot curve
        ax.set_xlabel("# training episodes")
        ax.set_ylabel("cumulative regret")
        ax.set_ylim(0, max(mrg+cl for _, mrg, cl in stats) * 1.2)
        ax.grid()

        # Ordering legends according to the prespecified ordering above
        handles, labels = ax.get_legend_handles_labels()
        hls_sorted = sorted(
            [(h, l) for h, l in zip(handles, labels)],
            key=lambda x: config_ord.index(x[1])
        )
        handles = [hl[0] for hl in hls_sorted]
        labels = [hl[1] for hl in hls_sorted]
        ax.legend(handles, labels)
        
        ax.set_title(f"Cumulative regret curve ({diff}; N={len(data[0])} per config)")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"cumulRegs_{diff}.png"))

    # Aggregate and visualize: learning curve & confusion matrices
    last_num_exs = max(results_confMat)
    for num_exs, agg_stats in results_confMat.items():
        for diff, per_diff in agg_stats.items():
            for exp_config, data in per_diff.items():
                feedStratT, semStratL = exp_config
                config_label = f"{semStratL}_{feedStratT}"

                # Compute aggregate metric: mean F1 score
                per_concept_Ps = [
                    [confMat[i,i] / confMat[:,i].sum() for i in range(len(confMat))]
                    for confMat in data["matrix"]
                ]
                per_concept_Rs = [
                    [confMat[i,i] for i in range(len(confMat))]
                    for confMat in data["matrix"]
                ]
                per_concept_F1s = [
                    # If both P & R are zero, F1 zero
                    [2*p*r/(p+r) if p+r>0 else 0.0 for p, r in zip(ps, rs)]
                    for ps, rs in zip(per_concept_Ps, per_concept_Rs)
                ]
                mean_F1s = [np.mean(F1s) for F1s in per_concept_F1s]

                # Collect data for plotting learning curve (by # examples used vs. mF1)
                if exp_config in results_learningCurve[diff]:
                    results_learningCurve[diff][exp_config].append((num_exs, mean_F1s))
                else:
                    results_learningCurve[diff][exp_config] = [(num_exs, mean_F1s)]

                if num_exs == last_num_exs:
                    # Draw confusion matrix
                    fig = plt.figure()
                    draw_matrix(
                        sum(data["matrix"]) / data["num_test_suites"],
                        data["concepts"], data["concepts"], fig.gca()    # Binary choice mode
                        # data["concepts"][:-1], data["concepts"], fig.gca()    # Multiple choice mode
                    )
                    plt.suptitle(f"Confusion matrix for ({diff}; N={len(data['matrix'])} per config)", fontsize=16)
                    plt.title(f"{config_label} agent", pad=18)
                    plt.tight_layout()
                    plt.savefig(os.path.join(cfg.paths.outputs_dir, f"confMat_{diff}_{config_label}.png"))

    # Aggregate and visualize: learning curve
    for diff, agg_stats in results_learningCurve.items():
        fig, ax = plt.subplots()

        for exp_config, data in agg_stats.items():
            feedStratT, semStratL = exp_config
            data = sorted(data)
            stats = [
                (i, np.mean(mF1s), 1.96 * np.std(mF1s)/np.sqrt(len(mF1s)))
                for i, mF1s in data
            ]

            # Plot mean curve
            curve = ax.plot(
                [i+1 for i, _, _ in stats],
                [mmF1 for _, mmF1, _ in stats],
                label=f"{semStratL}_{feedStratT}"
            )
            # Plot confidence intervals
            ax.fill_between(
                [i+1 for i, _, _ in stats],
                [mmF1-cl for _, mmF1, cl in stats],
                [mmF1+cl for _, mmF1, cl in stats],
                color=curve[0].get_color(), alpha=0.2
            )

        # Plot curve
        ax.set_xlabel("# training examples")
        ax.set_ylabel("mF1 score")
        ax.set_xlim(0, last_num_exs+10)
        ax.set_ylim(0, 1)
        ax.grid()

        # Ordering legends according to the prespecified ordering above
        handles, labels = ax.get_legend_handles_labels()
        hls_sorted = sorted(
            [(h, l) for h, l in zip(handles, labels)],
            key=lambda x: config_ord.index(x[1])
        )
        handles = [hl[0] for hl in hls_sorted]
        labels = [hl[1] for hl in hls_sorted]
        plt.legend(handles, labels)
        
        plt.title(f"Learning curve ({diff}; N={len(data[0][1])} per config)")
        plt.savefig(os.path.join(cfg.paths.outputs_dir, f"learningCurve_{diff}.png"))

    # Report final mF1 scores on terminal
    for diff, agg_stats in results_learningCurve.items():
        print("")
        logger.info(f"Final mean F1 scores ({diff}):")

        final_mF1s = {
            f"{semStratL}_{feedStratT}": sorted(data, reverse=True)[0][1]
            for (feedStratT, semStratL), data in agg_stats.items()
        }
        for cfg in sorted(final_mF1s, key=lambda x: config_ord.index(x)):
            logger.info("\t"+f"{cfg}: {float(np.mean(final_mF1s[cfg]))}")


if __name__ == "__main__":
    main()
