"""
For testing prediction: an interactive IterableDataset where the user is prompted to select
an image to process, either by entering file names or randomly, implemented as a subclass
of torch.utils.data.IterableDataset
"""
import readline

from torch.utils.data import IterableDataset

from ...utils.completer import DatasetImgsCompleter


class InteractiveDataset(IterableDataset):
    """
    At each next() call, ask for a user input for specifying which image file to read
    from a directory designated at __init__() and return in appropriate format.
    """
    def __init__(self, img_dir_path):
        self.img_dir_path = img_dir_path

        self.dcompleter = DatasetImgsCompleter()
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self.dcompleter.complete)
    
    def __iter__(self):
        print("")

        while True:
            while True:
                print(f"Sys> Choose an image to process")
                print("Sys> Enter 'r' for random selection, 'n' for skipping new image input")
                usr_in = input("U> ")
                print("")

                try:
                    if usr_in == "n":
                        print(f"Sys> Cancelled image selection")
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
            
            if img_f is None: break

            yield { "file_name": img_f }
