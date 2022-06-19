"""
Script for fine-grained grounding experiments; simulate natural interactions between
agent (learner) and user (teacher) with varying configurations
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from itl import ITLAgent
from itl.opts import parse_arguments
from .sim_user import SimulatedTeacher


if __name__ == "__main__":
    ...
