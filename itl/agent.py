"""
Outermost wrapper containing ITL agent API
"""
import readline
import rlcompleter

from .memory import LongTermMemoryModule
from .vision import VisionModule
from .vision.utils.completer import DatasetImgsCompleter
from .lang import LanguageModule
from .cognitive_reasoning import CognitiveReasonerModule
from .practical_reasoning import PracticalReasonerModule


TAB = "\t"  # For use in format strings

class ITLAgent:

    def __init__(self, opts):
        self.lt_mem = LongTermMemoryModule(opts)
        self.vision = VisionModule(opts)
        self.lang = LanguageModule(opts, lex=self.lt_mem.lexicon)
        self.cognitive = CognitiveReasonerModule(kb=self.lt_mem.knowledge_base)
        self.practical = PracticalReasonerModule()

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
            print("U> ", end="")
            usr_in = input()
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
        # Inform the language module of the visual context
        self.lang.situate(self.vision.vis_raw, self.vision.vis_scene)

        print("")
        print("Sys> Awaiting user input...")
        print("Sys> Enter 'n' for skipping language input")

        valid_input = False
        while not valid_input:
            print("U> ", end="")
            usr_in = input()
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
                print("Sys> Ungrammatical input, try again")
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
    
    def _update_belief(self):
        """Form beliefs based on visual and/or language input"""
        dialogue_state = self.lang.dialogue.export_state()

        if self.vision.vis_raw is None and len(dialogue_state["record"]) == 0:
            # No information whatsoever to form any sort of beliefs
            print("A> (Idling the moment away...)")
            return

        if self.vision.vis_scene is not None:
            vis_out, vis_lang_out = self.cognitive.sensemake(
                self.vision.vis_scene, dialogue_state, self.lang.lexicon
            )

            _, marginals_v, _, _ = vis_out

            # Organize sensemaking results by object, with category sorted by confidences
            results_v = {
                args[0]: { "cls": [], "att": [], "rel": [] }
                for pred, args in marginals_v if pred == "object"
            }
            for atom in marginals_v:
                pred, args = atom

                if pred == "object":
                    continue    # Already processed
                else:
                    cat_type, cat_ind = pred.split("_")
                    confidence = marginals_v[atom]

                    if cat_type != "rel":
                        results_v[args[0]][cat_type].append((
                            self.lang.lexicon.d2s[(int(cat_ind), cat_type)], confidence
                        ))
                    else:
                        results_v[args[0]][cat_type].append((
                            self.lang.lexicon.d2s[(int(cat_ind), cat_type)], args[1], confidence
                        ))
            
            # Report beliefs from vision only
            print(f"A> I am seeing these objects:")
            for oi, preds in results_v.items():
                print(f"A> {TAB}Object {oi}:")

                cls_preds = [f"{pr[0][0]} ({pr[1]:.2f})" for pr in preds['cls']]
                print(f"A> {TAB*2}class: {', '.join(cls_preds)}")

                att_preds = [f"{pr[0][0]} ({pr[1]:.2f})" for pr in preds['att']]
                print(f"A> {TAB*2}attribute: {', '.join(att_preds)}")

                rel_preds = [f"{pr[0][0]}[{pr[1]}] ({pr[2]:.2f})" for pr in preds['rel']]
                print(f"A> {TAB*2}relation: {', '.join(rel_preds)}")

                print("A>")
            
            if vis_lang_out is not None:
                _, _, _, mismatches = vis_lang_out

                ## TODO: Print surprisal reports to resolve_mismatch action ##
                if len(mismatches) > 0:
                    print("A> However, I was quite surprised to hear that:")
            
                    for m in mismatches:
                        is_positive = m[0]     # 'True' stands for positive statement

                        if is_positive:
                            # Positive statement (fact)
                            message = f"{m[2][2][0]} is (a/an) {m[1][0]}"
                        else:
                            # Negative statement (constraint)
                            negated = [
                                f"{m2[2][0]} is (a/an) {m1[0]}" for m1, m2 in zip(m[1], m[2])
                            ]
                            message = f"Not {{{' & '.join(negated)}}}"

                        print(f"A> {TAB}{message} (surprisal: {round(m[3], 3)})")
                    
                        self.practical.agenda.append(("resolve_mismatch", m))

    def _act(self):
        """Choose & execute actions to process agenda items"""
        for todo in reversed(self.practical.agenda):
            print(todo)
