import os
import imghdr
import random
import readline


class DatasetImgsCompleter():
    """Completer that only autocompletes image files under datasets/ directory"""

    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path

        self._all_imgs = [
            (root, [f for f in files if imghdr.what(os.path.join(root, f))])
            for root, _, files in os.walk(data_dir_path)
        ]
        self._all_imgs = [
            (root, sorted(files)) for (root, files) in self._all_imgs if len(files) > 0
        ]
        self._all_imgs_flat = sum([
            [os.path.join(r, f) for f in fs] for r, fs in self._all_imgs
        ], [])

        self.responses = None
    
    def __contains__(self, file):
        return file in self._all_imgs_flat
    
    def complete(self, _, state):
        if os.path.isdir(self.data_dir_path):
            if self.responses is None:
                text = readline.get_line_buffer()
                match_dirs_short = [(r, fs) for r, fs in self._all_imgs if r.startswith(text)]
                match_dirs_long = [(r, fs) for r, fs in self._all_imgs if text.startswith(r)]

                responses_short = [r+"/" for r, _ in match_dirs_short]
                responses_long = sum([
                    [f"{r}/{f}" for f in fs if f.startswith(text.strip(r))]
                    for r, fs in match_dirs_long
                ], [])
                self.responses = responses_short + responses_long
            
            if state < len(self.responses):
                return self.responses[state][readline.get_begidx():]
            else:
                # Traversed all, reset
                self.responses = None
                return None
        else:
            return None

    def sample(self):
        return random.sample(self._all_imgs_flat, 1)[0]
