"""
Outermost wrapper containing ITL agent API
"""
import readline
import rlcompleter

from .vision import VisionModule
from .vision.utils.completer import DatasetImgsCompleter


class ITLAgent():

    def __init__(self, opts):
        # self.lt_mem = LongTermMemory()
        self.vision = VisionModule(opts)
        # self.lang = LanguageModule()
        # self.cognitive = CognitiveReasoningModule()
        # self.practical = PracticalReasoningModule()

        self.dcompleter = DatasetImgsCompleter()
        readline.parse_and_bind("tab: complete")

    def __call__(self):
        """Main function: Kickstart infinite ITL agent loop"""
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
            vis_raw, vis_scene = self.vision.predict(img_f, visualize=True)

            # Inform the dialogue manager of the visual context
            # self.dm.situate(vis_raw, vis_scene)
        
        # Restore default completer
        readline.set_completer(rlcompleter.Completer().complete)
    
    def _lang_inp(self):
        """Language input prompt (from user)"""
        ...
    
    def _update_belief(self):
        """Form beliefs based on visual and/or language input"""
        ...
    
    def _act(self):
        """Choose & execute actions to process agenda items"""
        # for todo in reversed(self.agenda):
        #     print(todo)
        ...
