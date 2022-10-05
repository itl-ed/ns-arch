"""
This module implements custom extensions of detectron2 parts we need for our scene
graph generation task. Note that extensions of detectron2 components in this module
are not meant to be imported by other modules, but registered and invoked rather
indirectly via detectron2 YAML config files in ./configs directory.
"""
from .meta_learner import MetaLearner

__all__ = ["MetaLearner"]
