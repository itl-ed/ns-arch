import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from itl.vision import VisionModule
from itl.opts import parse_arguments


if __name__ == "__main__":
    opts = parse_arguments()
    vm = VisionModule(opts, initial_load=False)
    vm.dm.prepare_data()
