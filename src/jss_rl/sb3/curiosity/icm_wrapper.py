import gym

import torch

from functools import reduce
from typing import List, Dict

import numpy as np
import torch.nn as nn

from copy import copy
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
from torch.nn.functional import one_hot

from jss_rl.sb3.curiosity.icm_memory import IcmMemory
from jss_rl.sb3.util.torch_dense_sequential_model_builder import build_dense_sequential_network


class IntrinsicCuriosityModuleWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: VecEnvWrapper,

                 beta: float = 0.2,
                 eta: float = 1.0,
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

                 exploration_steps: int = None
                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        if not isinstance(venv.action_space, gym.spaces.Discrete):
            raise NotImplementedError

        self.exploration_steps = exploration_steps

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

        # self.venv.observation_space.shape does return weird shapes on some environments
        # (i.e. frozenlake -> () for a scalar value)
        # self._observation_dim = reduce((lambda x, y: x * y), o_shape)
        self._observation_dim = len(self.venv.reset()[0].ravel())

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

        # for intrinsic return info
        self._intrinsic_rewards = [np.array([]) for _ in range(self.venv.num_envs)]
        # for extrinsic return info
        self._extrinsic_rewards = [np.array([]) for _ in range(self.venv.num_envs)]

    def step_async(self, actions: np.ndarray) -> None:
        self._num_timesteps += self.venv.num_envs  # one step per env
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()

        intrinsic_rewards = np.zeros(self.venv.num_envs)

        if self.exploration_steps is not None and self._num_timesteps > self.exploration_steps:
            for i, done in enumerate(dones):
                self._extrinsic_rewards[i] = np.append(self._extrinsic_rewards[i], [rewards[i]])
                self._intrinsic_rewards[i] = np.append(self._intrinsic_rewards[i], [intrinsic_rewards[i]])
                infos[i]["extrinsic_reward"] = rewards[i]
                infos[i]['intrinsic_reward'] = intrinsic_rewards[i]
                if done:
                    infos[i]["extrinsic_return"] = self._extrinsic_rewards[i].sum()
                    infos[i]['intrinsic_return'] = self._intrinsic_rewards[i].sum()
                    infos[i]['total_return'] = infos[i]["extrinsic_return"] + infos[i]['intrinsic_return']
                    self._extrinsic_rewards[i] = np.array([])
                    self._intrinsic_rewards[i] = np.array([])
            return observations, rewards, dones, infos

        if self.prev_observations is not None and self.actions is not None:
            # iterate over envs in venv
            intrinsic_rewards = self._calc_intrinsic_reward(
                obs=self.prev_observations,
                actions=self.actions,
                next_obs=observations
            )
            # add prev_obs, action, obs tuples to memory (one tuple for each env in the venv)
            self.icm_memory.add_multiple_entries(
                prev_obs=self.prev_observations,
                actions=self.actions,
                obs=observations
            )

        for i, done in enumerate(dones):
            self._extrinsic_rewards[i] = np.append(self._extrinsic_rewards[i], [rewards[i]])
            self._intrinsic_rewards[i] = np.append(self._intrinsic_rewards[i], [intrinsic_rewards[i]])

            infos[i]["extrinsic_reward"] = rewards[i]
            infos[i]['intrinsic_reward'] = intrinsic_rewards[i]

            if done:
                infos[i]["extrinsic_return"] = self._extrinsic_rewards[i].sum()
                infos[i]['intrinsic_return'] = self._intrinsic_rewards[i].sum()
                infos[i]['total_return'] = infos[i]["extrinsic_return"] + infos[i]['intrinsic_return']

                self._extrinsic_rewards[i] = np.array([])
                self._intrinsic_rewards[i] = np.array([])

        augmented_reward = rewards + intrinsic_rewards

        if self._num_timesteps % self.postprocess_every_n_steps == 0:
            post_proc_info = self.postprocess_trajectory()
            for i, old_infos in enumerate(infos):
                infos[i] = {**old_infos, **post_proc_info}

        # override previous observations
        self.prev_observations = observations
        return observations, augmented_reward, dones, infos

    def _calc_intrinsic_reward(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray):
        with torch.no_grad():
            # Push both observations through feature net to get both phis.
            # sₜ ≙ obs
            # sₜ₊₁ ≙ next_obs

            # sₜ ⟼ Φ(sₜ)
            if self._observation_dim == 1:
                obs = np.array([np_array.ravel() for np_array in obs])
                next_obs = np.array([np_array.ravel() for np_array in next_obs])

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

        if self._observation_dim == 1:
            prev_obs = np.array([np_array.ravel() for np_array in prev_obs])
            obs = np.array([np_array.ravel() for np_array in obs])

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
        # zero grad before new step
        self._optimizer.zero_grad()
        # Backward pass and update
        loss.backward()
        self._optimizer.step()

        return {
            "icm_loss": loss.item()
        }

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()

        self.prev_observations = observations
        self.actions = None

        # reset trackers
        self._intrinsic_rewards = [np.array([]) for _ in range(self.venv.num_envs)]
        self._extrinsic_rewards = [np.array([]) for _ in range(self.venv.num_envs)]

        if self.clear_memory_on_reset:
            self.icm_memory = IcmMemory(
                capacity=self.memory_capacity,
                obs_shape=self.venv.observation_space.shape,
                action_shape=np.array(self.venv.action_space.sample()).shape
            )

        return observations


if __name__ == '__main__':
    from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3 import A2C

    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=1
    )

    budget = 10_000
    eval_episodes = 10

    cartpole_venv = VecMonitor(venv=venv)

    model1 = A2C('MlpPolicy', cartpole_venv, verbose=0, seed=773)

    model1.learn(total_timesteps=budget)
    mean_reward, std_reward = evaluate_policy(model1, cartpole_venv, n_eval_episodes=eval_episodes)
    print(f"without icm: {mean_reward=}, {std_reward=}")

    cartpole_venv.reset()
    cartpole_icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
        eta=0.15,
        exploration_steps=5_000
    )
    cartpole_icm_venv = VecMonitor(venv=cartpole_icm_venv)

    model2 = A2C('MlpPolicy', cartpole_icm_venv, verbose=0, seed=773)
    # model2.set_env(cartpole_venv)
    model2.learn(total_timesteps=budget)

    mean_reward, std_reward = evaluate_policy(model2, model2.get_env(), n_eval_episodes=eval_episodes)

    print(f"with icm: {mean_reward=}, {std_reward=}")
