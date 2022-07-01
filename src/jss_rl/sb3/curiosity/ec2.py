import itertools
import sys
from collections import deque
from copy import copy

import gym
import torch

from typing import Union, List, Dict

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvWrapper, VecEnv, VecEnvObs
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv

from jss_rl.sb3.util.moving_avarage import MovingAverage
from jss_rl.sb3.util.torch_dense_sequential_model_builder import _create_fc_net
from torch.nn.functional import one_hot


class EpisodicCuriosityModuleWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: Union[VecEnvWrapper, VecEnv, DummyVecEnv],
                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        self._num_timesteps = 0