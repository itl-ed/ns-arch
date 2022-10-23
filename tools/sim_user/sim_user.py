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
import numpy as np
from detectron2.structures import BoxMode


class SimulatedTeacher:
    
    def __init__(self, target_concepts, strat_feedback, test_set_size, seed):
        with open("datasets/tabletop/annotations.json") as data_file:
            self.data_annotation = json.load(data_file)
        with open("datasets/tabletop/metadata.json") as md_file:
            self.metadata = json.load(md_file)
        with open("tools/sim_user/tabletop_domain.json") as dom_file:
            self.domain_knowledge = {
                concept_string.split(".")[0].replace("_", " "): data
                for concept_string, data in json.load(dom_file).items()
            }
        
        self.data_by_img_id = {
            img["image_id"]: img for img in self.data_annotation
        }

        self.image_dir_prefix = "datasets/tabletop/images"

        # Target concepts to teach to agent; naturally controls ITL difficulty
        self.target_concepts = target_concepts

        # Indexing instances in images by concept in advance
        self.exemplars_cls = defaultdict(dict)
        self.exemplars_att = defaultdict(dict)
        self.exemplars_rel = defaultdict(dict)

        for img in self.data_annotation:
            indexing_per_img = {
                "cls": defaultdict(list),
                "att": defaultdict(list),
                "rel": defaultdict(list)
            }

            for oi, obj in enumerate(img["annotations"]):
                # Index by class
                for c in obj["classes"]:
                    indexing_per_img["cls"][c].append(oi)
                
                # Index by attribute
                for a in obj["attributes"]:
                    indexing_per_img["att"][a].append(oi)

                # Index by relation
                for rels in obj["relations"]:
                    for r in rels["relation"]:
                        indexing_per_img["rel"][r].append((oi, rels["object_id"]))

            # Collate by img and append
            for c, objs in indexing_per_img["cls"].items():
                c_name = self.metadata["classes"][c]
                self.exemplars_cls[c_name][img["image_id"]] = objs
            for a, objs in indexing_per_img["att"].items():
                a_name = self.metadata["attributes"][a]
                self.exemplars_att[a_name][img["image_id"]] = objs
            for r, obj_pairs in indexing_per_img["rel"].items():
                r_name = self.metadata["relations"][r]
                self.exemplars_rel[r_name][img["image_id"]] = obj_pairs
        
        self.exemplars_cls = dict(self.exemplars_cls)
        self.exemplars_att = dict(self.exemplars_att)
        self.exemplars_rel = dict(self.exemplars_rel)

        # Set seed
        random.seed(seed)

        # Split exemplars for training and testing per concept
        self.training_exemplars = {}; self.test_exemplars = {}
        for cat_type, concepts in target_concepts.items():
            if cat_type not in self.training_exemplars:
                self.training_exemplars[cat_type] = {}
            if cat_type not in self.test_exemplars:
                self.test_exemplars[cat_type] = {}

            for conc in concepts:
                all_exs = copy.deepcopy(list(
                    getattr(self, f"exemplars_{cat_type}")[conc].items()
                ))
                random.shuffle(all_exs)
                self.training_exemplars[cat_type][conc] = all_exs[:-test_set_size]
                self.test_exemplars[cat_type][conc] = all_exs[-test_set_size:]

        # History of ITL episode records
        self.episode_records = []

        # Pieces of generic constrastive knowledge taught across episodes
        self.taught_diffs = set()

        # Teacher's strategy on how to give feedback upon student's wrong answer
        # (provided the student has taken initiative for extended ITL interactions
        # by asking further questions after correct answer feedback)
        self.strat_feedback = strat_feedback

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
        sampled_instance = random.sample(instances, 1)[0]

        img_f = sampled_img["file_name"]
        instance_bbox = np.array(sampled_img["annotations"][sampled_instance]["bbox"])
        instance_bbox = BoxMode.convert(
            instance_bbox[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
        )[0]

        self.current_focus = (img_f, instance_bbox)

        # Temporary code for preparing 'cheat sheet', for checking whether logical
        # reasoners perform better if the agent's poor vision module's performance 
        # is replaced by oracle ground truths about object parts and their properties
        instance_parts = [
            r for r in sampled_img["annotations"][sampled_instance]["relations"]
            if "have.v.01" in [self.metadata["relations"][ri] for ri in r["relation"]]
        ]
        instance_parts = [
            sampled_img["annotations"][r["object_id"]] for r in instance_parts
        ]
        instance_parts = [
            (obj, [self.metadata["classes"][ci] for ci in obj["classes"]])
            for obj in instance_parts
        ]
        cheat_sheet = [
            (
                BoxMode.convert(
                    np.array(obj["bbox"])[None], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS
                )[0],
                classes[0].split(".")[0],
                [
                    self.metadata["attributes"][ai].split(".")[0] + \
                        "/" + classes[0].split(".")[0]
                    for ai in obj["attributes"]
                ]
            )
            for obj, classes in instance_parts
            if "bowl.n.01" in classes or "stem.n.03" in classes
        ]

        return {
            "v_usr_in": os.path.join(".", self.image_dir_prefix, img_f),
            "l_usr_in": "What is this?",
            "pointing": { "this": [instance_bbox] },
            "cheat_sheet": cheat_sheet
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
                if self.strat_feedback != "min" or is_novel_concept:
                    result_response["l_usr_in"] += f" This is a {concept_string}."
                    result_response["pointing"]["this"].append(self.current_focus[1])

                responses.append(result_response)
                
                # Generic difference between intended concept vs. incorrect answer
                # concept additionally provided if teacher strategy is 'greater'
                # than [maximal feedback]
                if self.strat_feedback == "max" and not is_novel_concept:
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
