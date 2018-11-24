"""
A genetic algorithm for Reinforcement Learning.
"""
#!/usr/bin/env python3
import inspect
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# very important, don't remove, otherwise pybullet cannot run (reasons are unknown)
import pybullet_envs
from .learning import LearningSession
from .models import FeedforwardPolicy, MLP, simple_mlp, nature_cnn
from .noise import NoiseSource, NoiseAdder, noise_seed
from .selection import truncation_selection, tournament_selection
from .util import make_session
