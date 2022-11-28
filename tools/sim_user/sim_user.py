"""
Simulated user for fine-grained grounding experiment suite; manage dialogue with
rule-based pattern matching -- no cognitive architecture ongoing within the user
"""
import os
import re
import json
import copy
import random
from collections import defaultdict

import inflect
import torch
from torchvision.ops import box_convert


class SimulatedTeacher:
    
    def __init__(self, cfg, target_concepts):
        tabletop_data_dir = os.path.join(cfg.paths.data_dir, "tabletop")
        with open(os.path.join(tabletop_data_dir, "annotations.json")) as data_file:
            self.data_annotation = json.load(data_file)
        with open(os.path.join(tabletop_data_dir, "metadata.json")) as md_file:
            self.metadata = json.load(md_file)
        with open(os.path.join(cfg.paths.root_dir, "tools/sim_user/tabletop_domain.json")) as dom_file:
            self.domain_knowledge = {
                concept_string.split(".")[0].replace("_", " "): data
                for concept_string, data in json.load(dom_file).items()
            }
        
        self.data_by_img_id = {
            img["image_id"]: img for img in self.data_annotation
        }

        self.image_dir_prefix = os.path.join(tabletop_data_dir, "images")

        # Target concepts to teach to agent; naturally controls ITL difficulty
        self.target_concepts = target_concepts

        # Concept instances, re-indexed by string name
        self.exemplars_cls = {
            self.metadata["classes_names"][int(ci)]: {
                int(k): v for k, v in insts.items()
            }
            for ci, insts in self.metadata["classes_instances"].items()
        }
        self.exemplars_att = {
            self.metadata["attributes_names"][int(ai)]: {
                int(k): v for k, v in insts.items()
            }
            for ai, insts in self.metadata["attributes_instances"].items()
        }
        self.exemplars_rel = {
            self.metadata["relations_names"][int(ci)]: {
                int(k): v for k, v in insts.items()
            }
            for ci, insts in self.metadata["relations_instances"].items()
        }

        # Set seed
        random.seed(cfg.seed)

        # Split exemplars for training and testing per concept
        self.training_exemplars = {}; self.test_exemplars = {}
        for conc_type, concepts in target_concepts.items():
            if conc_type not in self.training_exemplars:
                self.training_exemplars[conc_type] = {}
            if conc_type not in self.test_exemplars:
                self.test_exemplars[conc_type] = {}

            for conc in concepts:
                all_exs = copy.deepcopy(list(
                    getattr(self, f"exemplars_{conc_type}")[conc].items()
                ))
                random.shuffle(all_exs)
                self.training_exemplars[conc_type][conc] = all_exs[:-cfg.exp1.test_set_size]
                self.test_exemplars[conc_type][conc] = all_exs[-cfg.exp1.test_set_size:]

        # History of ITL episode records
        self.episode_records = []

        # Pieces of generic constrastive knowledge taught across episodes
        self.taught_diffs = set()

        # Teacher's strategy on how to give feedback upon student's wrong answer
        # (provided the student has taken initiative for extended ITL interactions
        # by asking further questions after correct answer feedback)
        self.strat_feedback = cfg.exp1.strat_feedback

    def initiate_episode(self):
        """
        Initiate an ITL episode by asking a what-question on a concept instance
        """
        # Fetch next target concept to test/teach this episode, then rotate the list
        self.current_target_concept = self.target_concepts["cls"].pop()
        self.target_concepts["cls"] = [self.current_target_concept] + self.target_concepts["cls"]

        concept_string = self.current_target_concept.split(".")[0]
        concept_string = concept_string.replace("_", " ")

        # Initialize episode record
        self.current_record = {
            "target_concept": concept_string,
            "answered_concept": None,
            "answer_correct": None,
            "number_of_exemplars": 0        # Exemplars used for learning
        }

        # Ideally, for situated (robotic) agent, the teacher would simply place an
        # exemplar of the target concept in front of the agent's vision sensor...
        # In this experiment, let's load an image containing an instance of the target
        # concept, and then present to the agent along with the question.
        img_cands = self.training_exemplars["cls"][self.current_target_concept]
        sampled_img, instances = img_cands.pop()
        sampled_img = self.data_by_img_id[sampled_img]
        sampled_instance = str(random.sample(instances, 1)[0])

        img_f = sampled_img["file_name"]
        instance_bbox = torch.tensor(sampled_img["annotations"][sampled_instance]["bbox"])
        instance_bbox = box_convert(instance_bbox[None], "xywh", "xyxy")[0].numpy()

        self.current_focus = (img_f, instance_bbox)

        # # Temporary code for preparing 'cheat sheet', for checking whether logical
        # # reasoners perform better if the agent's poor vision module's performance 
        # # is replaced by oracle ground truths about object parts and their properties
        # instance_parts = [
        #     r for r in sampled_img["annotations"][sampled_instance]["relations"]
        #     if "have.v.01" in [self.metadata["relations"][ri] for ri in r["relation"]]
        # ]
        # instance_parts = [
        #     sampled_img["annotations"][r["object_id"]] for r in instance_parts
        # ]
        # instance_parts = [
        #     (obj, [self.metadata["classes"][ci] for ci in obj["classes"]])
        #     for obj in instance_parts
        # ]
        # cheat_sheet = [
        #     (
        #         BoxMode.convert(
        #             np.array(obj["bbox"])[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
        #         )[0],
        #         classes[0].split(".")[0],
        #         [
        #             self.metadata["attributes"][ai].split(".")[0] + \
        #                 "/" + classes[0].split(".")[0]
        #             for ai in obj["attributes"]
        #         ]
        #     )
        #     for obj, classes in instance_parts
        #     if "bowl.n.01" in classes or "stem.n.03" in classes
        # ]

        return {
            "v_usr_in": os.path.join(".", self.image_dir_prefix, img_f),
            "l_usr_in": "What is this?",
            "pointing": { "this": [instance_bbox] },
            # "cheat_sheet": cheat_sheet
        }

    def react(self, agent_reaction):
        """ Rule-based pattern matching for handling agent responses """
        responses = []      # Return value containing response utterances

        concept_string = self.current_target_concept.split(".")[0]
        concept_string = concept_string.replace("_", " ")

        agent_utterances = [
            content for act_type, content in agent_reaction
            if act_type == "generate"
        ]

        if "I am not sure." in agent_utterances:
            # Agent answered it doesn't have any clue what the concept instance is;
            # provide correct label, even if taking minimalist strategy (after all,
            # learning cannot take place if we don't provide any)
            self.current_record["answered_concept"] = "N/A"
            self.current_record["answer_correct"] = False
            self.current_record["number_of_exemplars"] += 1

            responses.append({
                "v_usr_in": "n",
                "l_usr_in": f"This is a {concept_string}.",
                "pointing": { "this": [self.current_focus[1]] }
            })

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

                responses.append({
                    "v_usr_in": "n",
                    "l_usr_in": "Correct.",
                    "pointing": None
                })
            else:
                # Incorrect answer; reaction branches here depending on teacher's strategy
                self.current_record["answer_correct"] = False
                self.current_record["number_of_exemplars"] += 1

                # Minimal feedback; only let the agent know the answer is incorrect
                result_response = {
                    "v_usr_in": "n",
                    "l_usr_in": f"This is not a {answer_content}.",
                    "pointing": { "this": [self.current_focus[1]] }
                }

                # Correct label additionally provided if teacher strategy is 'greater' than
                # [minimal feedback] or the concept hasn't ever been taught
                taught_concepts = set(epi["target_concept"] for epi in self.episode_records)
                is_novel_concept = concept_string not in taught_concepts
                if self.strat_feedback != "minHelp" or is_novel_concept:
                    result_response["l_usr_in"] += f" This is a {concept_string}."
                    result_response["pointing"]["this"].append(self.current_focus[1])

                responses.append(result_response)
                
                # Generic difference between intended concept vs. incorrect answer
                # concept additionally provided if teacher strategy is 'greater'
                # than [maximal feedback]
                if self.strat_feedback == "maxHelp" and not is_novel_concept:
                    # Give generics only if not given previously, and if current target
                    # concept is not introduced for the first time
                    contrast_concepts = frozenset([concept_string, answer_content])
                    pluralize = inflect.engine().plural

                    if contrast_concepts not in self.taught_diffs:
                        target_props = self.domain_knowledge[concept_string]["part_property"]
                        target_props = {
                            (part, prop) for part, props in target_props.items() for prop in props
                        }
                        answer_props = self.domain_knowledge[answer_content]["part_property"]
                        answer_props = {
                            (part, prop) for part, props in answer_props.items() for prop in props
                        }

                        # For each of two directions of relative differences, synthesize
                        # appropriate constrastive generic explanations
                        target_props_diff = defaultdict(set)
                        target_subject = pluralize(concept_string.capitalize())
                        for part, prop in target_props - answer_props:
                            target_props_diff[part].add(prop)
                        for part, props in target_props_diff.items():
                            part_name = pluralize(part.split(".")[0])
                            part_descriptor = ", ".join(pr.split(".")[0] for pr in props)
                            generic = f"{target_subject} have {part_descriptor} {part_name}."
                            responses.append({
                                "v_usr_in": "n",
                                "l_usr_in": generic,
                                "pointing": None
                            })

                        answer_props_diff = defaultdict(set)
                        answer_subject = pluralize(answer_content.capitalize())
                        for part, prop in answer_props - target_props:
                            answer_props_diff[part].add(prop)
                        for part, props in answer_props_diff.items():
                            part_name = pluralize(part.split(".")[0])
                            part_descriptor = ", ".join(pr.split(".")[0] for pr in props)
                            generic = f"{answer_subject} have {part_descriptor} {part_name}."
                            responses.append({
                                "v_usr_in": "n",
                                "l_usr_in": generic,
                                "pointing": None
                            })

                        self.taught_diffs.add(contrast_concepts)

        else:
            raise NotImplementedError

        return responses
