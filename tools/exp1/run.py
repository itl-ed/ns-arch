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

    cumul_regret = 0; regret_curve = []; confMats_seq = []
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
            confMats_seq.append(ep["midterm_result"])

    with open(os.path.join(res_dir, f"cumulRegs_{tail}.csv"), "w") as out_csv:
        out_csv.write("Episode,Regret\n")
        for i, cumul_regret in enumerate(regret_curve):
            out_csv.write(f"{i},{cumul_regret}\n")

    for num_exs, data in confMats_seq:
        with open(os.path.join(res_dir, f"confMat{num_exs}_{tail}.csv"), "w") as out_csv:
            out_csv.write(",".join(concepts_ordered)+"\n")          # Binary mode
            # out_csv.write(",".join(concepts_ordered+["NA"])+"\n")   # Multiple choice mode
            for row in data / cfg.exp1.test_set_size:
                out_csv.write(",".join([str(d) for d in row])+"\n")


def midterm_test(agent, user, num_exs):
    # 'Mid-term exams' during the series of ITL interactions
    exam_result = defaultdict(lambda: defaultdict(int))
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
            for conc_test in user.test_exemplars["cls"]:
                if agent_answers[conc_test]:
                    concept_test_string = conc_test.split(".")[0]
                    concept_test_string = concept_test_string.replace("_", " ")
                    exam_result[concept_string][concept_test_string] += 1

            ## Multiple-choice, pick-one mode
            # test_input = {
            #     "v_usr_in": os.path.join(user.image_dir_prefix, img_f),
            #     "l_usr_in": f"What is this?",
            #     "pointing": { "this": [instance_bbox] }
            # }

            # agent_reaction = agent.loop(**test_input)
            # agent_utterances = [
            #     content for act_type, content in agent_reaction
            #     if act_type == "generate"
            # ]

            # if any(utt.startswith("This is") for utt in agent_utterances):
            #     answer_utt = [
            #         utt for utt in agent_utterances if utt.startswith("This is")
            #     ][0]
            #     answer_content = re.findall(r"This is a (.*)\.$", answer_utt)[0]
            #     exam_result[concept_string][answer_content] += 1
            # else:
            #     exam_result[concept_string]["NA"] += 1

    # Store exam result as confusion matrix
    C = len(test_problems)
    concepts_ordered = sorted(list(test_problems))      # Sort for consistent ordering

    data = np.zeros([C,C])      # Binary mode
    # data = np.zeros([C,C+1])    # Multiple choice mode
    for i in range(C):
        for j in range(C):
            conc_i = concepts_ordered[i]
            conc_j = concepts_ordered[j]

            data[i,j] = exam_result[conc_i][conc_j]
            # When multiple choice mode
            # data[i,-1] = exam_result[conc_i]["NA"]

    user.episode_records[-1]["midterm_result"] = (num_exs, data)

    # Return sequence of concepts for bookkeeping concept order
    return concepts_ordered


if __name__ == "__main__":
    main()
