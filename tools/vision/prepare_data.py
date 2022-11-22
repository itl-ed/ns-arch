import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import warnings
warnings.filterwarnings("ignore")

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf

from itl.vision.data import FewShotSGGDataModule


@hydra.main(config_path="../../itl/configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)

    dm = FewShotSGGDataModule(cfg)
    dm.prepare_data()


if __name__ == "__main__":
    main()
