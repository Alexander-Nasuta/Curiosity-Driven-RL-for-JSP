import itertools
import sys
import gym

import torch

from functools import reduce
from typing import List, Dict

import numpy as np
import torch.nn as nn

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
from torch.nn.functional import one_hot

from jss_rl.sb3.util.memory import Memory, IcmMemory


def build_dense_sequential_network(input_dim: int, output_dim: int, layers: List[int] = None, activation_function=None):
    if layers is None:
        layers = []
    if activation_function is None:
        activation_function = nn.Tanh()
    # all dims
    dims = np.array([input_dim, *layers, output_dim])
    # in out pairs
    l_in_out = [torch.nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims, dims[1:])]
    # linear, act function pairs
    l_in_out = zip(
        l_in_out,
        [activation_function] * len(l_in_out)
    )
    # flatten (list of tupels -> list)
    l_in_out = itertools.chain.from_iterable(l_in_out)
    return nn.Sequential(*l_in_out)


class IntrinsicCuriosityModuleWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: VecEnvWrapper,

                 beta: float = 0.2,
                 eta: float = 10.0,
                 lr: float = 1e-3,

                 device: str = 'cpu',

                 feature_dim: int = 288,
                 feature_net_hiddens: List[int] = None,
                 feature_net_activation=nn.Tanh(),
                 inverse_feature_net_hiddens: List[int] = None,
                 inverse_feature_net_activation=nn.Tanh(),
                 forward_fcnet_net_hiddens: List[int] = None,
                 forward_fcnet_net_activation=nn.Tanh(),

                 postprocess_every_n_steps: int = 100,
                 postprocess_sample_size: int = 100,
                 memory_capacity: int = 10_000,
                 shuffle_memory_samples: bool = True,
                 clear_memory_on_reset=False,
                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        if not isinstance(venv.action_space, gym.spaces.Discrete):
            raise NotImplementedError

        if feature_net_hiddens is None:
            feature_net_hiddens = [64, 64]

        if inverse_feature_net_hiddens is None:
            inverse_feature_net_hiddens = [64, 64]

        if forward_fcnet_net_hiddens is None:
            forward_fcnet_net_hiddens = [64, 64]

        self.device = device

        self._num_timesteps = 0
        self.postprocess_every_n_steps = postprocess_every_n_steps

        # memory / history
        self.prev_observations = venv.reset()
        self.actions = None
        self.memory_capacity = memory_capacity
        self.clear_memory_on_reset = clear_memory_on_reset
        self.icm_memory = IcmMemory(
            capacity=self.memory_capacity,
            obs_shape=venv.observation_space.shape,
            action_shape=np.array(venv.action_space.sample()).shape
        )
        self.shuffle_memory_samples = shuffle_memory_samples
        self.postprocess_sample_size = postprocess_sample_size

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
        self._observation_dim = reduce((lambda x, y: x * y), venv.observation_space.shape)

        # s ⟼ Φ
        self._curiosity_feature_net = build_dense_sequential_network(
            input_dim=self._observation_dim,
            output_dim=self.feature_dim,
            activation_function=feature_net_activation,
            layers=feature_net_hiddens
        )

        # Φ(sₜ), Φ(sₜ₊₁) ⟼ aₜᵖʳᵉᵈ
        self._curiosity_inverse_fcnet = build_dense_sequential_network(
            input_dim=2 * self.feature_dim,
            output_dim=self.action_dim,
            activation_function=inverse_feature_net_activation,
            layers=inverse_feature_net_hiddens
        )

        # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
        self._curiosity_forward_fcnet = build_dense_sequential_network(
            input_dim=self.feature_dim + self.action_dim,
            output_dim=self.feature_dim,
            activation_function=forward_fcnet_net_activation,
            layers=forward_fcnet_net_hiddens
        )

        # see ray impl 'get_exploration_optimizer'
        feature_params = list(self._curiosity_feature_net.parameters())
        inverse_params = list(self._curiosity_inverse_fcnet.parameters())
        forward_params = list(self._curiosity_forward_fcnet.parameters())

        self._curiosity_feature_net = self._curiosity_feature_net.to(self.device)
        self._curiosity_inverse_fcnet = self._curiosity_inverse_fcnet.to(self.device)
        self._curiosity_forward_fcnet = self._curiosity_forward_fcnet.to(self.device)

        self._optimizer = torch.optim.Adam(forward_params + inverse_params + feature_params, lr=self.lr)

        # for intrisic return info
        self._intrinsic_rewards = [[]] * venv.num_envs  # empty list for every env

    def step_async(self, actions: np.ndarray) -> None:
        self._num_timesteps += 1
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()

        # print(f"before: {rewards}")
        intrinsic_rewards = np.zeros(self.venv.num_envs)
        if self.prev_observations is not None and self.actions is not None:
            # iterate over envs in venv
            for i, (prev_o, a, o) in enumerate(zip(self.prev_observations, self.actions, observations)):
                a = np.array(a)
                self.icm_memory.add(prev_obs=prev_o, action=a, obs=o)
                i_rew = self._calc_intrinsic_reward(obs=prev_o, actions=a, next_obs=o)
                intrinsic_rewards[i] = i_rew
                infos[i]['intrinsic_reward'] = i_rew
                self._intrinsic_rewards[i].append(i_rew)
                rewards[i] += i_rew

        for i, done in enumerate(dones):
            if not done:
                continue
            infos[i]['intrinsic_return'] = np.sum(np.array(self._intrinsic_rewards[i]))
            if "extrinsic_return" in infos[i].keys():
                infos[i]['total_return'] = infos[i]['intrinsic_return'] + infos[i]['extrinsic_return']
            self._intrinsic_rewards[i] = []

        if self._num_timesteps % self.postprocess_every_n_steps == 0:
            post_proc_info = self.postprocess_trajectory()
            for i, old_infos in enumerate(infos):
                infos[i] = {**old_infos, **post_proc_info}

        return observations, rewards, dones, infos

    def _calc_intrinsic_reward(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray):
        with torch.no_grad():
            # Push both observations through feature net to get both phis.
            # sₜ ≙ obs
            # sₜ₊₁ ≙ next_obs

            # sₜ ⟼ Φ(sₜ)
            phi = self._curiosity_feature_net(
                torch.from_numpy(obs.astype(np.float32))
            )
            # sₜ₊₁ ⟼ Φ(sₜ₊₁)
            next_phi = self._curiosity_feature_net(
                torch.from_numpy(next_obs.astype(np.float32))
            )

            # Predict next phi with forward model.
            # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
            #
            # Equation 4
            #
            # Φ(sₜ₊₁)ᵖʳᵉᵈ  =  f(Φ(sₜ),aₜ;θᵢ)

            actions_tensor = (
                torch.from_numpy(
                    actions
                ).long().to(self.device)
            )
            actions_one_hot = one_hot(
                actions_tensor, self.action_space.n
            ).float()
            predicted_next_phi = self._curiosity_forward_fcnet(torch.cat([phi, actions_one_hot], dim=-1))

            # Forward loss term (predicted phi', given phi and action vs actually
            # observed phi').
            #
            #  Equation 5
            #
            #     ⎛                  ⎞     1                           2
            #  Lբ ⎜ Φ(sₜ), Φ(sₜ₊₁)ᵖʳᵉᵈ⎟ =  ───  ║ Φ(sₜ₊₁)ᵖʳᵉᵈ - Φ(sₜ)  ║₂
            #     ⎝                  ⎠     2
            #
            forward_l2_norm_squared = 0.5 * torch.sum(
                torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
            )

            return self.eta * forward_l2_norm_squared.detach().cpu().numpy()

    def postprocess_trajectory(self) -> Dict:
        """
                Calculates phi values (obs, obs', and predicted obs') and ri.

                Also calculates forward and inverse losses and updates the curiosity
                module on the provided batch using our optimizer.
                """
        prev_obs, actions, obs = self.icm_memory.sample(
            shuffle=self.shuffle_memory_samples,
            batch_size=self.postprocess_sample_size
        )

        # Push both observations through feature net to get both phis.
        # sₜ ≙ obs
        # sₜ₊₁ ≙ next_obs

        # sₜ ⟼ Φ(sₜ)
        phi = self._curiosity_feature_net(
            torch.from_numpy(prev_obs.astype(np.float32))
        )
        # sₜ₊₁ ⟼ Φ(sₜ₊₁)
        next_phi = self._curiosity_feature_net(
            torch.from_numpy(obs.astype(np.float32))
        )

        # Predict next phi with forward model.
        # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
        #
        # Equation 4
        #
        # Φ(sₜ₊₁)ᵖʳᵉᵈ  =  f(Φ(sₜ),aₜ;θᵢ)

        actions_tensor = (
            torch.from_numpy(
                actions
            ).long().to(self.device)
        )
        actions_one_hot = one_hot(
            actions_tensor, self.action_space.n
        ).float()
        predicted_next_phi = self._curiosity_forward_fcnet(torch.cat([phi, actions_one_hot], dim=-1))

        # Forward loss term (predicted phi', given phi and action vs actually
        # observed phi').
        #
        #  Equation 5
        #
        #     ⎛                  ⎞     1                           2
        #  Lբ ⎜ Φ(sₜ), Φ(sₜ₊₁)ᵖʳᵉᵈ⎟ =  ───  ║ Φ(sₜ₊₁)ᵖʳᵉᵈ - Φ(sₜ)  ║₂
        #     ⎝                  ⎠     2
        #
        forward_l2_norm_squared = 0.5 * torch.sum(
            torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
        )

        forward_loss = torch.mean(forward_l2_norm_squared)  # Lբ

        # Inverse loss term (prediced action that led from phi to phi' vs
        # actual action taken).
        phi_cat_next_phi = torch.cat([phi, next_phi], dim=-1)
        #
        # Φ(sₜ), Φ(sₜ₊₁) ⟼ aₜᵖʳᵉᵈ
        #
        # Equation 2
        #
        # aₜᵖʳᵉᵈ  =  f(Φ(sₜ), Φ(sₜ₊₁);θբ)
        #
        dist_inputs = self._curiosity_inverse_fcnet(phi_cat_next_phi)
        actions_dist = torch.distributions.Categorical(logits=dist_inputs)

        # Neg log(p); p=probability of observed action given the inverse-NN
        # predicted action distribution.

        inverse_loss = -actions_dist.log_prob(actions_tensor)
        inverse_loss = torch.mean(inverse_loss)

        # Calculate the ICM loss.
        #
        # Equation 7
        #
        # min  (1-β) Lᵢ + β Lբ
        #
        # NOTE: the agents policy network is optimised seperatedly in this implementation
        loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss

        # optimizer step
        #
        # Backward pass and update
        loss.backward()
        self._optimizer.step()
        # zero grad before new step
        self._optimizer.zero_grad()

        return {
            "icm_loss": loss.item()
        }

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()

        self.prev_observations = observations
        self.actions = None

        # reset intrinsic reward tracker
        self._intrinsic_rewards = [[]] * self.venv.num_envs

        if self.clear_memory_on_reset:
            self.icm_memory = IcmMemory(
                capacity=self.memory_capacity,
                obs_shape=self.venv.observation_space.shape,
                action_shape=np.array(self.venv.action_space.sample()).shape
            )

        return observations


if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3 import A2C

    import copy

    env_id = "CartPole-v1"

    cartpole_venv = make_vec_env(
        env_id=env_id,
        env_kwargs={},
        n_envs=4
    )

    budget = 5_000
    eval_episodes = 10

    icm_venv = IntrinsicCuriosityModuleWrapper(venv=cartpole_venv)
    _, r, _, info = icm_venv.step(np.array([1] * 4))
    print(r)

    model1 = A2C('MlpPolicy', cartpole_venv, verbose=0)
    model2 = copy.deepcopy(model1)
    model1.learn(total_timesteps=budget)

    mean_reward, std_reward = evaluate_policy(model1, icm_venv, n_eval_episodes=eval_episodes)

    print(f"without icm: {mean_reward=}, {std_reward=}")
    sys.exit(1)

    cartpole_venv.reset()
    cartpole_venv = IntrinsicCuriosityModuleWrapper(venv=cartpole_venv)

    model2.set_env(cartpole_venv)
    model2.learn(total_timesteps=budget)

    mean_reward, std_reward = evaluate_policy(model2, model2.get_env(), n_eval_episodes=eval_episodes)

    print(f"with icm: {mean_reward=}, {std_reward=}")
