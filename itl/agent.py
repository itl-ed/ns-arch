"""
Outermost wrapper containing ITL agent API
"""
import readline
import rlcompleter
from collections import OrderedDict

from detectron2.structures import BoxMode

from .memory import LongTermMemoryModule
from .vision import VisionModule
from .vision.utils.completer import DatasetImgsCompleter
from .lang import LanguageModule
from .cognitive_reasoning import CognitiveReasonerModule
from .practical_reasoning import PracticalReasonerModule


TAB = "\t"  # For use in format strings

class ITLAgent:

    def __init__(self, opts):
        # Initialize component modules
        self.lt_mem = LongTermMemoryModule()
        self.vision = VisionModule(opts)
        self.lang = LanguageModule(opts, lex=self.lt_mem.lexicon)
        self.cognitive = CognitiveReasonerModule(kb=self.lt_mem.kb)
        self.practical = PracticalReasonerModule()

        # Initialize empty lexicon with concepts in visual module
        self.lt_mem.lexicon.fill_from_vision(self.vision.predicates)

        # Image file selection CUI
        self.dcompleter = DatasetImgsCompleter()
        readline.parse_and_bind("tab: complete")

    def __call__(self):
        """Main function: Kickstart infinite ITL agent loop"""
        print(f"Sys> At any point, enter 'exit' to quit")

        while True:
            self._vis_inp()
            self._lang_inp()
            self._update_belief()
            self._act()
    
    def _vis_inp(self):
        """Image input prompt (Choosing from dataset for now)"""
        # Register autocompleter for REPL
        readline.set_completer(self.dcompleter.complete)

        print("")

        while True:

            print(f"Sys> Choose an image to process")
            print("Sys> Enter 'r' for random selection, 'n' for skipping new image input")
            usr_in = input("U> ")
            print("")

            try:
                if usr_in == "n":
                    print(f"Sys> Skipped image selection")
                    img_f = None
                    break
                elif usr_in == "r":
                    img_f = self.dcompleter.sample()
                elif usr_in == "exit":
                    print(f"Sys> Terminating...")
                    quit()
                else:
                    if usr_in not in self.dcompleter:
                        raise ValueError(f"Image file {usr_in} does not exist")
                    img_f = usr_in

            except ValueError as e:
                print(f"Sys> {e}, try again")

            else:
                print(f"Sys> Selected image file: {img_f}")
                break

        if img_f is not None:
            # Run visual inference on designated image, and store raw image & generated
            # scene graph
            self.vision.predict(img_f, visualize=True)
        
        # Restore default completer
        readline.set_completer(rlcompleter.Completer().complete)
    
    def _lang_inp(self):
        """Language input prompt (from user)"""
        if self.vision.updated:
            # Inform the language module of the visual context
            self.lang.situate(self.vision.vis_raw, self.vision.vis_scene)
            self.cognitive.refresh()

        print("")
        print("Sys> Awaiting user input...")
        print("Sys> Enter 'n' for skipping language input")

        valid_input = False
        while not valid_input:
            usr_in = input("U> ")
            print("")

            # Understand the user input in the context of the dialogue
            try:
                if usr_in == "n":
                    print(f"Sys> Skipped language input")
                    break
                elif usr_in == "exit":
                    print(f"Sys> Terminating...")
                    quit()
                else:
                    self.lang.understand(usr_in, self.practical.agenda)

            except IndexError as e:
                print(f"Sys> Ungrammatical input or IndexError: {e.args}")
                continue
            except ValueError as e:
                print(f"Sys> {e.args[0]}")
                continue
            except NotImplementedError:
                print("Sys> Sorry, can't handle the input sentence (yet)")
                continue

            else:
                valid_input = True
                break

        if self.vision.vis_scene is not None:
            # If a new entity is registered as a result of understanding the latest
            # input, re-run vision module to update with new predictions for it
            if len(self.lang.dialogue.referents["env"]) > len(self.vision.vis_scene):
                bboxes = [
                    {
                        "bbox": ent["bbox"],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "objectness_scores": self.vision.vis_scene[name]["pred_objectness"]
                            if name in self.vision.vis_scene else None
                    }
                    for name, ent in self.lang.dialogue.referents["env"].items()
                ]

                # Predict on latest raw data stored
                vis_raw_bgr = self.vision.vis_raw[:, :, [2,1,0]]
                self.vision.predict(vis_raw_bgr, bboxes=bboxes, visualize=True)
    
    def _update_belief(self):
        """Form beliefs based on visual and/or language input"""

        if self.vision.vis_raw is None and len(self.lang.dialogue.record) == 0:
            # No information whatsoever to form any sort of beliefs
            print("A> (Idling the moment away...)")
            return

        if self.vision.vis_scene is not None:
            # Make final conclusions via sensemaking
            self.cognitive.sensemake(self.vision, self.lang, self.practical)

            models_v, _, _ = self.cognitive.concl_vis
            marginals_v = models_v.marginals()

            # Organize sensemaking results by object, with category sorted by confidences
            results_v = {
                atom.args[0][0]: { "obj": score, "cls": [], "att": [], "rel": [] }
                for atom, score in marginals_v.items() if atom.name == "object"
            }
            for atom in marginals_v:
                pred = atom.name
                args = atom.args

                if pred == "object":
                    continue    # Already processed
                else:
                    cat_type, cat_ind = pred.split("_")
                    confidence = marginals_v[atom]

                    if cat_type != "rel":
                        results_v[args[0][0]][cat_type].append((
                            self.lang.lexicon.d2s[(int(cat_ind), cat_type)], confidence
                        ))
                    else:
                        results_v[args[0][0]][cat_type].append((
                            self.lang.lexicon.d2s[(int(cat_ind), cat_type)], args[1][0], confidence
                        ))
            results_v = OrderedDict(sorted(results_v.items(), key=lambda v: int(v[0].strip("o"))))
            
            # Report beliefs from vision only
            print(f"A>")
            print(f"A> I am seeing these objects:")
            for oi, preds in results_v.items():
                print(f"A> {TAB}Object {oi} ({preds['obj']:.2f}):")

                cls_preds = [f"{pr[0][0]} ({pr[1]:.2f})" for pr in preds['cls']]
                print(f"A> {TAB*2}class: {', '.join(cls_preds)}")

                att_preds = [f"{pr[0][0]} ({pr[1]:.2f})" for pr in preds['att']]
                print(f"A> {TAB*2}attribute: {', '.join(att_preds)}")

                rel_preds = [f"{pr[0][0]}[{pr[1]}] ({pr[2]:.2f})" for pr in preds['rel']]
                print(f"A> {TAB*2}relation: {', '.join(rel_preds)}")

                print("A>")
            
            self.vision.reshow_pred()

    def _act(self):
        """
        Just eagerly try to resolve each item in agenda as much as possible, generating
        and performing actions until no more agenda items can be resolved for now. I
        wonder if we'll ever need a more sophisticated mechanism than this simple, greedy
        method for a good while?
        """
        while True:
            resolved_items = []
            for i, todo in enumerate(self.practical.agenda):
                todo_state, todo_args = todo

                # Check if this item can be resolved at this stage and if so, obtain
                # appropriate plan (sequence of actions) for resolving the item
                plan = self.practical.obtain_plan(todo_state)

                if plan is not None:
                    act_seq, prereqs = plan

                    # Test prerequisites
                    prereqs_pass = True
                    for pr in prereqs:
                        pr_method = pr["prereq_method"].extract(self)
                        pr_args = pr["prereq_args_getter"](todo_args)
                        if type(pr_args) == tuple:
                            pr_args = tuple(a.extract(self) for a in pr_args)
                        else:
                            pr_args = (pr_args.extract(self),)

                        if pr_method(*pr_args) != pr["prereq_value"]:
                            prereqs_pass = False
                            break
                    if not prereqs_pass: continue

                    # Perform plan actions
                    for action in act_seq:
                        act_method = action["action_method"].extract(self)
                        act_args = action["action_args_getter"](todo_args)
                        if type(act_args) == tuple:
                            act_args = tuple(a.extract(self) for a in act_args)
                        else:
                            act_args = (act_args.extract(self),)

                        act_method(*act_args)
                    
                    resolved_items.append(i)

            if len(resolved_items) == 0:
                # No resolvable agenda item any more; break
                break
            else:
                # Check off resolved agenda item
                resolved_items.reverse()
                for i in resolved_items:
                    del self.practical.agenda[i]
