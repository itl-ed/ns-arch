"""
Simulated user for fine-grained grounding experiment suite; manage dialogue with
rule-based pattern matching -- no cognitive architecture ongoing within the user
"""
import os
import re
import json
import copy
import random

import numpy as np
from detectron2.structures import BoxMode


class SimulatedTeacher:
    
    def __init__(self, target_concepts, strat_feedback, test_set_size, seed):
        with open("datasets/tabletop/annotations.json") as data_file:
            self.data_annotation = json.load(data_file)
        with open("datasets/tabletop/metadata.json") as md_file:
            self.metadata = json.load(md_file)
        with open("tools/sim_user/tabletop_domain.json") as dom_file:
            self.domain_knowledge = json.load(dom_file)

        self.image_dir_prefix = "datasets/tabletop/images"

        # Target concepts to teach to agent; naturally controls ITL difficulty
        self.target_concepts = target_concepts

        # Fetch in advance set of images containing any instance of the target
        # concepts
        self.img_candidates = {
            conc: [
                (
                    img,
                    [self.metadata["classes"].index(conc) in obj["classes"]
                        for obj in img["annotations"]]
                ) for img in self.data_annotation
            ]
            for conc in self.target_concepts
        }
        self.img_candidates = {
            conc: [
                (img, [oi for oi, is_inst in enumerate(is_insts) if is_inst])
                for (img, is_insts) in imgs if any(is_insts)
            ]
            for conc, imgs in self.img_candidates.items()
        }

        # Exemplars for training
        self.training_exemplars = {
            conc: copy.deepcopy(imgs[:-test_set_size])
            for conc, imgs in self.img_candidates.items()
        }
        random.seed(seed)
        for imgs in self.training_exemplars.values():
            random.shuffle(imgs)

        # Exemplars for testing
        self.test_exemplars = {
            conc: copy.deepcopy(imgs[-test_set_size:])
            for conc, imgs in self.img_candidates.items()
        }

        # History of ITL episode records
        self.episode_records = []

        # Teacher's strategy on how to give feedback upon student's wrong answer
        # (provided the student has taken initiative for extended ITL interactions
        # by asking further questions after correct answer feedback)
        self.strat_feedback = strat_feedback

    def initiate_episode(self):
        """
        Initiate an ITL episode by asking a what-question on a concept instance
        """
        # Fetch next target concept to test/teach this episode, then rotate the list
        self.current_target_concept = self.target_concepts.pop()
        self.target_concepts = [self.current_target_concept] + self.target_concepts

        concept_string = self.current_target_concept.split(".")[0]
        concept_string = concept_string.replace("_", " ")

        # Initialize episode record
        self.current_record = {
            "target_concept": concept_string,
            "answered_concept": None,
            "answer_correct": None,
            "number_of_exemplars": 0       # Exemplars used for learning
        }

        # Ideally, for situated (robotic) agent, the teacher would simply place an
        # exemplar of the target concept in front of the agent's vision sensor...
        # In this experiment, let's load an image containing an instance of the target
        # concept, and then present to the agent along with the question.
        img_cands = self.training_exemplars[self.current_target_concept]
        sampled_img, instances = img_cands.pop()
        sampled_instance = random.sample(instances, 1)[0]

        img_f = sampled_img["file_name"]
        instance_bbox = np.array(sampled_img["annotations"][sampled_instance]["bbox"])
        instance_bbox = BoxMode.convert(
            instance_bbox[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
        )[0]

        self.current_focus = (img_f, instance_bbox)

        return {
            "v_usr_in": os.path.join(self.image_dir_prefix, img_f),
            "l_usr_in": "What is this?",
            "pointing": { "this": [instance_bbox] }
        }

    def react(self, agent_reaction):
        """ Rule-based pattern matching for handling agent responses """
        concept_string = self.current_target_concept.split(".")[0]
        concept_string = concept_string.replace("_", " ")

        agent_utterances = [
            content for act_type, content in agent_reaction
            if act_type == "generate"
        ]

        if "I am not sure." in agent_utterances:
            # Agent answered it doesn't have any clue what the concept instance is;
            # provide correct label, even if taking minimalist strategy (after all,
            # learning cannot take place if we don't provide any!)
            self.current_record["answered_concept"] = "N/A"
            self.current_record["answer_correct"] = False
            self.current_record["number_of_exemplars"] += 1

            response = {
                "v_usr_in": "n",
                "l_usr_in": f"This is a {concept_string}.",
                "pointing": { "this": [self.current_focus[1]] }
            }

        elif any(utt.startswith("This is") for utt in agent_utterances):
            # Agent provided an answer what the instance is
            answer_utt = [
                utt for utt in agent_utterances if utt.startswith("This is")
            ][0]
            answer_content = re.findall(r"This is a (.*)\.$", answer_utt)[0]

            self.current_record["answered_concept"] = answer_content

            if concept_string == answer_content:
                # Correct answer
                self.current_record["answer_correct"] = True
                self.current_record["number_of_exemplars"] += 0

                response = {
                    "v_usr_in": "n",
                    "l_usr_in": "Correct.",
                    "pointing": None
                }
            else:
                # Incorrect answer; reaction branches here depending on teacher's strategy
                self.current_record["answer_correct"] = False
                self.current_record["number_of_exemplars"] += 1

                # Minimal feedback; only let the agent know the answer is incorrect
                response = {
                    "v_usr_in": "n",
                    "l_usr_in": f"This is not a {answer_content}.",
                    "pointing": { "this": [self.current_focus[1]] }
                }

                # Additional labelling provided if teacher strategy is 'greater' than [minimal
                # feedback] or the concept hasn't ever been taught
                taught_concepts = set(epi["target_concept"] for epi in self.episode_records)
                is_novel_concept = concept_string not in taught_concepts
                if self.strat_feedback != "min" or is_novel_concept:
                    response["l_usr_in"] += f" This is a {concept_string}."
                    response["pointing"]["this"].append(self.current_focus[1])

        else:
            raise NotImplementedError

        return response
