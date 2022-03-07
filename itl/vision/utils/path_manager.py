"""
Registers a custom path handler for handling checkpoint paths with prefix "wandb://",
while making sure handlers used by detectron2 are properly registered as well
"""
import os

import wandb
from dotenv import load_dotenv
from iopath.common.file_io import PathManager as PathManagerBase
from iopath.common.file_io import PathHandler, HTTPURLHandler, get_cache_dir
from detectron2.utils.file_io import Detectron2Handler

__all__ = ["PathManager"]


PathManager = PathManagerBase()

class WandbHandler(PathHandler):
    """
    Resolve anything that's hosted as W&B stored files.
    """
    
    PREFIX = "wandb://"

    def __init__(self):
        super().__init__()
        load_dotenv("wandb.env")

    def _get_supported_prefixes(self):
        return [self.PREFIX]
    
    def _get_local_path(self, path, **kwargs):
        # Recognize W&B run file path
        path = path.strip(self.PREFIX).split("/")
        assert len(path) >= 4, \
            "W&B checkpoint path should be in form of " \
            "wandb://{entity}/{project}/{run_id}/{ckpt_file}"
        entity, project, run_id, *ckpt_file = path
        run = wandb.Api().run(f"{entity}/{project}/{run_id}")

        # Download ckpt file if not exists
        local_dir_path = os.path.join(
            get_cache_dir(), entity, project, run.name
        )
        os.makedirs(local_dir_path, exist_ok=True)
        run.file("/".join(ckpt_file)).download(root=local_dir_path, replace=True)
        
        return os.path.join(local_dir_path, *ckpt_file)
    
    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)

PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(Detectron2Handler())
PathManager.register_handler(WandbHandler())
