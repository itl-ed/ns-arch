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
import warnings
warnings.filterwarnings("ignore")

import tqdm
import torch
import numpy as np
from detectron2.structures import BoxMode

from itl import ITLAgent
from itl.opts import parse_arguments
from tools.sim_user import SimulatedTeacher


TAB = "\t"

if __name__ == "__main__":
    opts = parse_arguments()

    if opts.exp1_difficulty == "base":
        target_concepts = [
            "brandy_glass.n.*",
            "banana.n.01",
            "wine_bottle.n.01"
        ]
    elif opts.exp1_difficulty == "easy":
        target_concepts = [
            "brandy_glass.n.*",
            "burgundy_glass.n.*",
            "champagne_coupe.n.*"
        ]
    else:
        raise NotImplementedError

    # Number of episodes = Episode per concept * Number of concepts
    num_eps = opts.exp1_num_episodes * len(target_concepts)

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
        agent_reaction = agent.loop(**user_init)

        # Continue interaction until agent answers with "OK."
        while ("generate", "OK.") not in agent_reaction:
            user_reaction = user.react(agent_reaction)
            agent_reaction = agent.loop(**user_reaction)
        
        # End of episode, push record to history
        user.episode_records.append(user.current_record)

    res_dir = os.path.join(opts.output_dir_path, "exp1_res")

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
    for conc, imgs in user.test_exemplars.items():
        concept_string = conc.split(".")[0]
        concept_string = concept_string.replace("_", " ")

        for img, instances in tqdm.tqdm(imgs, total=len(imgs)):
            instance = instances[0]

            img_f = img["file_name"]
            instance_bbox = np.array(img["annotations"][instance]["bbox"])
            instance_bbox = BoxMode.convert(
                instance_bbox[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
            )[0]

            for conc_test in user.test_exemplars:
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

    # Plot exam result as confusion matrix
    C = len(exam_result)
    data = np.zeros([C,C])
    concepts_ordered = list(exam_result)

    with open(os.path.join(res_dir, f"confMat_{tail}.csv"), "w") as out_csv:
        out_csv.write(str(opts.exp1_test_set_size)+"\n")
        out_csv.write(",".join(concepts_ordered)+"\n")
        for i in range(C):
            for j in range(C):
                conc_i = concepts_ordered[i]
                conc_j = concepts_ordered[j]

                data[i,j] = exam_result[conc_i][conc_j] / opts.exp1_test_set_size
            out_csv.write(",".join([str(d) for d in data[i]])+"\n")

    # Cluster analysis of positive/negative exemplars
    from torch.utils.tensorboard import SummaryWriter

    fvecs_to_label = defaultdict(set); fvecs_imgs = {}
    for concept, (ex_vecs, ex_imgs) in agent.lt_mem.exemplars.pos_exs.items():
        for v, im in list(zip(ex_vecs, ex_imgs)):
            fvecs_to_label[tuple(v)].add((concept, "pos"))
            fvecs_imgs[tuple(v)] = im
    for concept, (ex_vecs, ex_imgs) in agent.lt_mem.exemplars.neg_exs.items():
        for v, im in list(zip(ex_vecs, ex_imgs)):
            fvecs_to_label[tuple(v)].add((concept, "neg"))
            fvecs_imgs[tuple(v)] = im

    concepts_ordered = \
        set(agent.lt_mem.exemplars.pos_exs) | set(agent.lt_mem.exemplars.neg_exs)
    concepts_ordered = list(concepts_ordered)
    metadata_ordered = [agent.lt_mem.lexicon.d2s[c][0][0] for c in concepts_ordered]

    fvecs_ordered = []
    labels_ordered = []
    imgs_ordered = []
    for v, labels in fvecs_to_label.items():
        fvecs_ordered.append(v)
        label_ex = ["n/a"] * len(concepts_ordered)
        for c, l in labels:
            label_ex[concepts_ordered.index(c)] = l
        labels_ordered.append(label_ex)
        imgs_ordered.append(np.transpose(fvecs_imgs[v], [2,0,1]))

    fvecs_ordered = np.array(fvecs_ordered)
    imgs_ordered = torch.tensor(np.stack(imgs_ordered)) / 255

    for i, c in enumerate(concepts_ordered):
        pos_exs = [
            v for v, labels in fvecs_to_label.items() if (c, "pos") in labels
        ]
        neg_exs = [
            v for v, labels in fvecs_to_label.items() if (c, "neg") in labels
        ]

        if len(pos_exs) > 0:
            pos_proto_vec = np.array([
                v for v, labels in fvecs_to_label.items() if (c, "pos") in labels
            ]).mean(axis=0, keepdims=True)
            fvecs_ordered = np.concatenate([fvecs_ordered, pos_proto_vec])
            pos_proto_label = ["n/a"] * len(concepts_ordered)
            pos_proto_label[i] = "pos_proto"
            labels_ordered.append(pos_proto_label)
            imgs_ordered = torch.cat(
                [imgs_ordered, torch.ones(imgs_ordered.shape[1:])[None]]
            )

        if len(neg_exs) > 0:
            neg_proto_vec = np.array([
                v for v, labels in fvecs_to_label.items() if (c, "neg") in labels
            ]).mean(axis=0, keepdims=True)
            fvecs_ordered = np.concatenate([fvecs_ordered, neg_proto_vec])
            neg_proto_label = ["n/a"] * len(concepts_ordered)
            neg_proto_label[i] = "neg_proto"
            labels_ordered.append(neg_proto_label)
            imgs_ordered = torch.cat(
                [imgs_ordered, torch.ones(imgs_ordered.shape[1:])[None]]
            )

    writer = SummaryWriter(os.path.join(res_dir, f"analysis_{tail}"))

    # Instance feature vectors
    writer.add_embedding(
        fvecs_ordered,
        metadata=labels_ordered,
        metadata_header=metadata_ordered,
        label_img=imgs_ordered,
        tag="Exemplar vector inspection"
    )
