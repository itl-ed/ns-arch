"""
Script for fine-grained grounding experiments; simulate natural interactions between
agent (learner) and user (teacher) with varying configurations
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import uuid
from collections import defaultdict
import logging
import warnings
warnings.filterwarnings("ignore")

import tqdm
import hydra
import torch
import numpy as np
from torchvision.ops import box_convert
from omegaconf import OmegaConf

from itl import ITLAgent
from tools.sim_user import SimulatedTeacher

logger = logging.getLogger(__name__)


TAB = "\t"

OmegaConf.register_new_resolver(
    "randid", lambda: str(uuid.uuid4())[:6]
)
@hydra.main(config_path="../../itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    if cfg.exp1.difficulty == "nonFine":
        target_concepts = {
            "cls": [
                "banana.n.01",
                "wine_bottle.n.01",
                "brandy_glass.n.*"
            ]
        }
    elif cfg.exp1.difficulty == "fineEasy":
        target_concepts = {
            "cls": [
                "champagne_coupe.n.*",
                "burgundy_glass.n.*",
                "brandy_glass.n.*"
            ]
        }
    elif cfg.exp1.difficulty == "fineHard":
        target_concepts = {
            "cls": [
                "champagne_coupe.n.*",
                "burgundy_glass.n.*",
                "brandy_glass.n.*",
                "martini_glass.n.*",
                "bordeaux_glass.n.*"
            ]
        }
    else:
        raise NotImplementedError

    # Set up agent & user
    agent = ITLAgent(cfg)
    user = SimulatedTeacher(cfg, target_concepts)

    # Turn off UI pop up on predict
    agent.vis_ui_on = False

    # Experiment name suffix
    tail = f"{cfg.exp1.difficulty}_" \
        f"{cfg.exp1.strat_feedback}_" \
        f"{cfg.agent.strat_generic}_" \
        f"{cfg.seed}"

    # Train until certain number of mistakes have been made (thus same number of 
    # examples were used for training), configured as cfg.exp1.num_examples
    i = 0; num_exs_used = 0
    while num_exs_used < cfg.exp1.num_examples:
        i += 1
        sys_msg = f"Episode {i} (# exs: {num_exs_used})"
        logger.info("Sys> " + ("*" * (len(sys_msg)+8)))
        logger.info(f"Sys> **  {sys_msg}  **")
        logger.info("Sys> " + ("*" * (len(sys_msg)+8)))
        # Each single ITL episode is initiated by the teacher, aiming to test and confer
        # knowledge on one of the target concepts specified
        user_init = user.initiate_episode()
        agent_reactions = agent.loop(**user_init)

        # Continue interaction until agent answers with "OK." to every user reaction
        while any(ar != ("generate", "OK.") for ar in agent_reactions):
            user_reactions = user.react([
                ar for ar in agent_reactions if ar != ("generate", "OK.")
            ])
            agent_reactions = sum([
                agent.loop(**ur) for ur in user_reactions
            ], [])

        # End of episode, push record to history
        user.episode_records.append(user.current_record)

        # Update total # of examples used for training
        new_num_exs_used = sum(ep["number_of_examples"] for ep in user.episode_records)

        # Run performance tests on test batch every once in a while
        if new_num_exs_used > num_exs_used:
            # Only if num_exs_used increased; if same as prev, no learning has happened,
            # thus no point in re-taking midterm test
            if new_num_exs_used % cfg.exp1.test_interval == 0:
                # Every cfg.exp1.test_interval examples used
                concepts_ordered = midterm_test(agent, user, new_num_exs_used)
        
        num_exs_used = new_num_exs_used

    res_dir = os.path.join(cfg.paths.outputs_dir, "exp1_res")
    os.makedirs(res_dir, exist_ok=True)

    # Summarize ITL interaction records stored in the simulated user object
    logger.info("**************")
    logger.info("Sys> Experiment finished. Result:")

    cumul_regret = 0; regret_curve = []; midterm_results = []
    for i, ep in enumerate(user.episode_records):
        if ep["answer_correct"]:
            logger.info(f"Sys> {TAB}Episode {i+1}: Correct")
        else:
            cumul_regret += 1
            answer = ep["answered_concept"]
            ground_truth = ep["target_concept"]
            logger.info(f"Sys> {TAB}Episode {i+1}: Wrong")
            logger.info(f"Sys> {TAB*2}Answered: {answer} vs. Correct: {ground_truth}")
        
        regret_curve.append(cumul_regret)

        if "midterm_result" in ep:
            midterm_results.append(ep["midterm_result"])

    with open(os.path.join(res_dir, f"cumulRegs_{tail}.csv"), "w") as out_csv:
        out_csv.write("Episode,Regret\n")
        for i, cumul_regret in enumerate(regret_curve):
            out_csv.write(f"{i},{cumul_regret}\n")

    mAP_curve = []
    for num_exs, mAP, conf_mat in midterm_results:
        mAP_curve.append((num_exs, mAP))

        with open(os.path.join(res_dir, f"confMat{num_exs}_{tail}.csv"), "w") as out_csv:
            out_csv.write(",".join(concepts_ordered)+"\n")
            for row in conf_mat / cfg.exp1.test_set_size:
                out_csv.write(",".join([str(d) for d in row])+"\n")
    
    with open(os.path.join(res_dir, f"mAPs_{tail}.csv"), "w") as out_csv:
        out_csv.write("num_exs,mAP\n")
        for num_exs, mAP in mAP_curve:
            out_csv.write(f"{num_exs},{mAP}\n")

def midterm_test(agent, user, num_exs):
    # 'Mid-term exams' during the series of ITL interactions
    exam_result = defaultdict(list)
    test_problems = {
        conc.split(".")[0].replace("_", " "): imgs
        for conc, imgs in user.test_exemplars["cls"].items()
    }
    for concept_string, imgs in test_problems.items():
        for img, instances in tqdm.tqdm(imgs, total=len(imgs)):
            img = user.data_annotation[img]
            img_f = img["file_name"]

            instance = str(instances[0])
            instance_bbox = torch.tensor(img["annotations"][instance]["bbox"])
            instance_bbox = box_convert(instance_bbox[None], "xywh", "xyxy")[0].numpy()

            # Binary concept testing mode
            agent_answers = agent.test_binary(
                os.path.join(user.image_dir_prefix, img_f),
                instance_bbox,
                user.target_concepts["cls"]
            )
            for conc_ans, score in agent_answers.items():
                concept_ans_string = conc_ans.split(".")[0]
                concept_ans_string = concept_ans_string.replace("_", " ")
                exam_result[concept_string].append((concept_ans_string, score))

    # Compute and store mAP score, best-F1 score (across possible threshold values)
    # and confusion matrix (at F1-best threshold) from the collected exam result
    C = len(test_problems)
    concepts_ordered = sorted(list(test_problems))      # Sort for consistent ordering

    # Obtaining mAP (mean average precision) score
    # Collect results and sort (decreasing) by confidence score per answers,
    # to obtain per-concept P/R curve
    sorted_preds = defaultdict(list)
    for conc_truth, answers in exam_result.items():
        for conc_ans, score in answers:
            sorted_preds[conc_ans].append((score, conc_ans==conc_truth))
    sorted_preds = {
        conc_ans: sorted(preds, reverse=True, key=lambda x: x[0])
        for conc_ans, preds in sorted_preds.items()
    }

    # (Interpolated) Precision-recall curves, AP scores, mAP score
    APs = []
    for preds in sorted_preds.values():
        # Original (jagged) precision-recall curve
        cumul_TPs = np.cumsum([is_TP for _, is_TP in preds])
        pr_curve = [
            (true_pos/agent.cfg.exp1.test_set_size, true_pos/(all_pos+1))
            for all_pos, true_pos in zip(range(len(preds)), cumul_TPs)
        ]

        # Interpolated precision-recall curve
        pr_curve_interp = [
            (float(ir), float(ip))
            for ir, ip in zip(
                [r for r, _ in pr_curve],
                np.maximum.accumulate([p for _, p in pr_curve[::-1]])[::-1]
            )
        ]
        # Compute AP value as AUC of the interpolated curve; sum of boxes
        # spanning until unique recall points
        AP = 0.0; r_last = 0.0; p_current = pr_curve_interp[0][1]
        for i, (r, p) in enumerate(pr_curve_interp):
            if i == len(pr_curve_interp)-1:
                # Reached end of recall axis; should increment AP
                increment_AP = True
            else:
                # Increment AP if reached a "cliff edge" on the curve
                r_not_same = i==0 or (r != pr_curve_interp[i-1][0])
                p_will_drop = p > pr_curve_interp[i+1][1]
                increment_AP = r_not_same and p_will_drop

            if increment_AP:
                # Increment AP by area, then update r_last by left-end of new box
                AP += (r-r_last) * p_current
                r_last = r

            if i < len(pr_curve_interp)-1 and pr_curve_interp[i+1][1]==p:
                # Dropping precision reached next plateau
                p_current = pr_curve_interp[i+1][1]
        
        APs.append(AP)

    # Mean AP across concepts being tested
    mAP = sum(APs) / len(APs)

    # Obtaining confusion matrix at threshold 0.15
    conf_mat = np.zeros([C,C])
    for i in range(C):
        for j in range(C):
            conc_i = concepts_ordered[i]
            conc_j = concepts_ordered[j]

            conf_mat[i,j] = sum(
                score >= 0.15 for conc, score in exam_result[conc_i]
                if conc==conc_j
            )

    user.episode_records[-1]["midterm_result"] = (num_exs, mAP, conf_mat)

    # Return sequence of concepts for bookkeeping concept order
    return concepts_ordered


if __name__ == "__main__":
    main()
