import itertools
import sys
import torch

from functools import reduce
from typing import List

import numpy as np
import torch.nn as nn

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn


def build_dense_sequential_network(input_dim: int, output_dim: int, layers: List[int] = None, activation_function=None):
    if layers is None:
        layers = []
    if activation_function is None:
        activation_function = nn.Tanh()
    # all dims
    dims = np.array([input_dim, *layers, output_dim])
    # in out pairs
    layer_in_out = zip(dims, dims[1:])
    # to nn.Linear, activation function
    layer_in_out = [[nn.Linear(input_dim, out_dim), activation_function] for in_dim, out_dim in layer_in_out]
    # flatten
    layers_in_out = itertools.chain(*layer_in_out)
    return nn.Sequential(*layers_in_out)


class IntrinsicCuriosityModuleWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: VecEnvWrapper,
                 feature_dim: int = 288,
                 beta: float = 0.2,
                 eta: float = 10.0,
                 lr: float = 1e-3,
                 device: str = 'cpu',
                 curiosity_feature_net_arch: List[int] = None,
                 inverse_curiosity_feature_net_arch: List[int] = None,
                 curiosity_forward_fcnet_net_arch: List[int] = None,
                 postprocess_every_n_steps: int = 500
                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        if curiosity_feature_net_arch is None:
            curiosity_feature_net_arch = [64, 64]

        if inverse_curiosity_feature_net_arch is None:
            inverse_curiosity_feature_net_arch = [64, 64]

        if curiosity_forward_fcnet_net_arch is None:
            curiosity_forward_fcnet_net_arch = [64, 64]

        self.device = device

        self._num_timesteps = 0
        self.postprocess_every_n_steps = postprocess_every_n_steps

        # memory / history
        self.history = []
        self.prev_observations = venv.reset()
        self.actions = None

        # parameters
        self.feature_dim = feature_dim
        self.action_dim = venv.action_space.n

        # hyper parameters
        self.beta = beta  # β
        self.eta = eta  # η
        self.lr = lr

        # neural networks
        # ᵖʳᵉᵈ means predicted or 'hat' in paper notation

        # s
        self._curiosity_observation_dim = reduce((lambda x, y: x * y), venv.observation_space.shape)

        # s ⟼ Φ
        self._curiosity_feature_net = build_dense_sequential_network(
            input_dim=self._curiosity_observation_dim,
            output_dim=self.feature_dim,
            layers=curiosity_feature_net_arch
        )

        # Φ(sₜ), Φ(sₜ₊₁) ⟼ aₜᵖʳᵉᵈ
        self._curiosity_inverse_fcnet = build_dense_sequential_network(
            input_dim=2 * self.feature_dim,
            output_dim=self.action_dim,
            layers=inverse_curiosity_feature_net_arch
        )

        # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
        self._curiosity_forward_fcnet = build_dense_sequential_network(
            input_dim=self.feature_dim + self.action_dim,
            output_dim=self.feature_dim,
            layers=curiosity_forward_fcnet_net_arch
        )

        # see ray impl 'get_exploration_optimizer'
        feature_params = list(self._curiosity_feature_net.parameters())
        inverse_params = list(self._curiosity_inverse_fcnet.parameters())
        forward_params = list(self._curiosity_forward_fcnet.parameters())

        self._curiosity_feature_net = self._curiosity_feature_net.to(self.device)
        self._curiosity_inverse_fcnet = self._curiosity_inverse_fcnet.to(self.device)
        self._curiosity_forward_fcnet = self._curiosity_forward_fcnet.to(self.device)

        self._optimizer = torch.optim.Adam(forward_params + inverse_params + feature_params, lr=self.lr)

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        return observations, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()
        # self.postprocess_trajectory(history_batch=self.history)
        # self.prev_observations = observations
        # self.history = []
        return observations
