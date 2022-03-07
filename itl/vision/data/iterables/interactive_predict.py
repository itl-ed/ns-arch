"""
For testing prediction: an interactive IterableDataset where the user is prompted to select
an image to process, either by entering file names or randomly, implemented as a subclass
of torch.utils.data.IterableDataset
"""
import os
import random
import readline

from torch.utils.data import IterableDataset


class InteractiveDataset(IterableDataset):
    """
    At each next() call, ask for a user input for specifying which image file to read
    from a directory designated at __init__() and return in appropriate format.
    """
    def __init__(self, img_dir_path):
        self.img_dir_path = img_dir_path

        self.images_in_dir = [
            img
            for img in os.listdir(self.img_dir_path)
            if os.path.isfile(os.path.join(self.img_dir_path, img))
        ]

        # Register autocompleter for REPL
        def _completer(text, state):
            options = [img for img in self.images_in_dir if img.startswith(text)]
            if state < len(options):
                return options[state]
            else:
                return None

        readline.parse_and_bind("tab: complete")
        readline.set_completer(_completer)
    
    def __iter__(self):
        print("")

        while True:
            while True:
                print(f"Sys> Choose an image to process ({len(self.images_in_dir)} files in total)")
                print("Sys> Enter 'r' for random selection, 'n' for skipping new image input")
                usr_in = input("U> ")
                print("")

                try:
                    if usr_in == "n":
                        print(f"Sys> Cancelled image selection")
                        img = None
                        break
                    elif usr_in == "r":
                        img = random.sample(self.images_in_dir, 1)[0]
                    else:
                        if usr_in not in self.images_in_dir:
                            raise ValueError(f"Sys> Image file {usr_in} does not exist, try again")
                        img = usr_in

                except ValueError as e:
                    print(e)

                else:
                    print(f"Sys> Selected image file: {img}")
                    break
            
            if img is None: break

            yield { "file_name": os.path.join(self.img_dir_path, img) }
