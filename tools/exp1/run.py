"""
Script for fine-grained grounding experiments; simulate natural interactions between
agent (learner) and user (teacher) with varying configurations
"""
import re
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import tqdm
import numpy as np
from detectron2.structures import BoxMode

from itl import ITLAgent
from itl.opts import parse_arguments
from tools.sim_user import SimulatedTeacher


TAB = "\t"

if __name__ == "__main__":
    opts = parse_arguments()

    if opts.exp1_difficulty == "base":
        target_concepts = {
            "cls": [
                "banana.n.01",
                "wine_bottle.n.01",
                "brandy_glass.n.*"
            ]
        }
    elif opts.exp1_difficulty == "easy":
        target_concepts = {
            "cls": [
                "champagne_coupe.n.*",
                "burgundy_glass.n.*",
                "brandy_glass.n.*"
            ]
        }
    else:
        raise NotImplementedError

    # Number of episodes = Episode per concept * Number of concepts
    num_eps = opts.exp1_num_episodes * len(target_concepts["cls"])

    # Set up agent & user
    agent = ITLAgent(opts)
    user = SimulatedTeacher(
        target_concepts=target_concepts,
        strat_feedback=opts.exp1_strat_feedback,
        test_set_size=opts.exp1_test_set_size,
        seed=opts.exp1_random_seed
    )

    # Turn off UI pop up on predict
    agent.vis_ui_on = False

    # Experiment name suffix
    tail = f"{opts.exp1_difficulty}_" \
        f"{opts.exp1_strat_feedback}_" \
        f"{opts.strat_mismatch}_" \
        f"{opts.exp1_random_seed}"

    for i in tqdm.tqdm(range(num_eps), total=num_eps):
        print("")
        print(f"Sys> Episode {i+1}")
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

    res_dir = os.path.join(opts.output_dir_path, "exp1_res")
    os.makedirs(res_dir, exist_ok=True)

    with open(os.path.join(res_dir, f"curve_{tail}.csv"), "w") as out_csv:
        # Summarize ITL interaction records stored in the simulated user object
        print("")
        print("Sys> Experiment finished. Result:")

        out_csv.write("Episode,Regret\n")

        cumul_regret = 0
        for i, ep in enumerate(user.episode_records):
            if ep["answer_correct"]:
                print(f"Sys> {TAB}Episode {i+1}: Correct")
            else:
                cumul_regret += 1
                answer = ep["answered_concept"]
                ground_truth = ep["target_concept"]
                print(f"Sys> {TAB}Episode {i+1}: Wrong")
                print(f"Sys> {TAB*2}Answered: {answer} vs. Correct: {ground_truth}")

            out_csv.write(f"{i},{cumul_regret}\n")

    # Final 'exam' after the series of ITL interactions
    exam_result = defaultdict(lambda: defaultdict(int))
    for conc, imgs in user.test_exemplars["cls"].items():
        concept_string = conc.split(".")[0]
        concept_string = concept_string.replace("_", " ")

        for img, instances in tqdm.tqdm(imgs, total=len(imgs)):
            img = user.data_annotation[img]
            img_f = img["file_name"]

            instance = instances[0]
            instance_bbox = np.array(img["annotations"][instance]["bbox"])
            instance_bbox = BoxMode.convert(
                instance_bbox[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
            )[0]

            # Binary concept testing mode
            for conc_test in user.test_exemplars["cls"]:
                concept_test_string = conc_test.split(".")[0]
                concept_test_string = concept_test_string.replace("_", " ")

                test_input = {
                    "v_usr_in": os.path.join(user.image_dir_prefix, img_f),
                    "l_usr_in": f"Is this a {concept_test_string}?",
                    "pointing": { "this": [instance_bbox] }
                }

                agent_reaction = agent.loop(**test_input)

                agent_utterances = [
                    content for act_type, content in agent_reaction
                    if act_type == "generate"
                ]

                if any(utt.startswith("This is a ") for utt in agent_utterances):
                    # Agent provided an answer what the instance is
                    exam_result[concept_string][concept_test_string] += 1
                
                elif any(utt.startswith("This is not a ") for utt in agent_utterances):
                    # Agent provided an answer what the instance is
                    pass
                
                else:
                    raise NotImplementedError

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
    C = len(exam_result)
    data = np.zeros([C,C])      # Binary mode
    # data = np.zeros([C,C+1])    # Multiple choice mode
    concepts_ordered = list(exam_result)

    with open(os.path.join(res_dir, f"confMat_{tail}.csv"), "w") as out_csv:
        out_csv.write(str(opts.exp1_test_set_size)+"\n")
        out_csv.write(",".join(concepts_ordered)+"\n")          # Binary mode
        # out_csv.write(",".join(concepts_ordered+["NA"])+"\n")   # Multiple choice mode
        for i in range(C):
            for j in range(C):
                conc_i = concepts_ordered[i]
                conc_j = concepts_ordered[j]

                data[i,j] = exam_result[conc_i][conc_j] / opts.exp1_test_set_size
                # When multiple choice mode
                # data[i,-1] = exam_result[conc_i]["NA"] / opts.exp1_test_set_size

            out_csv.write(",".join([str(d) for d in data[i]])+"\n")
