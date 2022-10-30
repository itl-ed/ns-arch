import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import warnings
warnings.filterwarnings("ignore")

import hydra
from omegaconf import OmegaConf

from itl.vision import VisionModule


@hydra.main(config_path="../../itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    vision = VisionModule(cfg)
    vision.train()


if __name__ == "__main__":
    main()
