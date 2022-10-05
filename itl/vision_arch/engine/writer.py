"""
Our custom writer class for obtaining events from detectron2 EventStorage
and logging metrics
"""
import os
from collections import defaultdict

import yaml
import wandb
from dotenv import find_dotenv, load_dotenv
from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.events import (
    EventWriter,
    TensorboardXWriter,
    get_event_storage
)
from detectron2.evaluation.testing import flatten_results_dict


class WandbWriter(EventWriter):
    """
    Write all scalars to the wandb run via WandbLogger
    """
    def __init__(self, model, window_size=20, resume_id=None):
        """
        Args:
            model: torch.nn.Module; PyTorch model instance
            window_size: int; scalars will be median-smoothed by this window size
            resume_id: str (optional); W&B run id for resuming training
        """
        self.model = model
        self.window_size = window_size
        self.last_write = defaultdict(lambda: -1)

        load_dotenv(find_dotenv(raise_error_if_not_found=True))
        exp_dir_path = os.path.join(self.model.cfg.OUTPUT_DIR, self.model.exp_name)
        os.makedirs(exp_dir_path, exist_ok=True)

        wb_kwargs = {
            "name": self.model.exp_name,
            "dir": exp_dir_path,
            "config": yaml.safe_load(self.model.cfg.dump())
        }
        if resume_id is not None:
            wb_kwargs["resume"] = "must"
            wb_kwargs["id"] = resume_id

        wandb.init(**wb_kwargs)
        wandb.watch(self.model, log="all", log_graph=False)

    def write(self):
        storage = get_event_storage()
        to_log = { "global_step": self.model.global_step }

        for k, (v, iter) \
            in storage.latest_with_smoothing_hint(self.window_size).items():

            if iter > self.last_write[k]:
                to_log[k] = v
                self.last_write[k] = iter

        if len(storage._vis_data) >= 1:
            to_log["training_samples"] = [
                wandb.Image(img, caption=img_name)
                for img_name, img, _ in storage._vis_data
            ]
            
            storage.clear_images()

        wandb.log(to_log, step=self.model.global_step)

    def close(self):
        """Clean-up"""
        wandb.unwatch(self.model)


class TBXWriter(TensorboardXWriter):
    """
    Extension of TensorboardXWriter such that it complies with the logging convention
    used by our WandbWriter
    """
    def __init__(self, model, window_size=20, **kwargs):
        """
        Args:
            model: torch.nn.Module; PyTorch model instance
            window_size: int; scalars will be median-smoothed by this window size
            kwargs: other arguments passed to tensorboard.SummaryWriter()
        """
        self.model = model
        self.window_size = window_size
        self.last_write = defaultdict(lambda: -1)

        exp_dir_path = os.path.join(
            self.model.cfg.OUTPUT_DIR,
            self.model.exp_name,
            "tbx"
        )

        cfg_dump = yaml.safe_load(self.model.cfg.dump())
        cfg_dump = {k: str(v) for k, v in flatten_results_dict(cfg_dump).items()}

        self.writer = SummaryWriter(log_dir=exp_dir_path, **kwargs)
        self.writer.add_hparams(cfg_dump, {}, run_name=self.model.exp_name)

    def write(self):
        storage = get_event_storage()

        for k, (v, iter) \
            in storage.latest_with_smoothing_hint(self.window_size).items():

            if iter > self.last_write[k]:
                self.writer.add_scalar(k, v, self.model.global_step)
                self.last_write[k] = iter

        if len(storage._vis_data) >= 1:
            for img_name, img, _ in storage._vis_data:
                img = img.transpose(2, 0, 1)
                self.writer.add_image(img_name, img, self.model.global_step)
            storage.clear_images()
