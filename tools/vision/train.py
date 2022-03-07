import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from itl.vision import VisionModule
from itl.opts import parse_arguments


if __name__ == "__main__":
    os.environ["FVCORE_CACHE"] = ...

    opts = parse_arguments()
    vm = VisionModule(opts)
    vm.train(exp_name=opts.exp_name, resume=opts.resume_training)
