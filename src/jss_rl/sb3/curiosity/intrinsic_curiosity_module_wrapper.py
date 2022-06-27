import sys
from collections import deque

import gym

import torch

from functools import reduce
from typing import List, Dict, Union

import numpy as np
import torch.nn as nn

from copy import copy
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnv
from torch.nn.functional import one_hot

from jss_rl.sb3.curiosity.icm_memory import IcmMemory
from jss_rl.sb3.util.torch_dense_sequential_model_builder import build_dense_sequential_network, _create_fc_net


class IntrinsicCuriosityModuleWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: Union[VecEnvWrapper, VecEnv, DummyVecEnv],

                 beta: float = 0.2,
                 eta: float = 1.0,
                 lr: float = 1e-3,

                 device: str = 'cpu',

                 feature_dim: int = 288,
                 feature_net_hiddens: List[int] = None,
                 feature_net_activation=nn.ReLU(),
                 inverse_feature_net_hiddens: List[int] = None,
                 inverse_feature_net_activation=nn.ReLU(),
                 forward_fcnet_net_hiddens: List[int] = None,
                 forward_fcnet_net_activation=nn.ReLU(),

                 postprocess_every_n_steps: int = 100,
                 postprocess_sample_size: int = 100,
                 memory_capacity: int = 100,
                 shuffle_memory_samples: bool = False,
                 clear_memory_on_reset=False,
                 clear_memory_after_n_episodes: int = None,

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
        self._num_timesteps_since_last_postprocessing = 0
        self._num_episodes = 0
        self._num_episodes_since_cleared_icm_memory = 0
        self.postprocess_every_n_steps = postprocess_every_n_steps

        # s
        if isinstance(self.venv.observation_space, gym.spaces.Box):
            self.o_shape = self.venv.observation_space.shape
            self._observation_dim = reduce((lambda x, y: x * y), self.o_shape)

        elif isinstance(self.venv.observation_space, gym.spaces.Discrete):
            self.o_shape = (self.venv.observation_space.n,)
            self._observation_dim = self.venv.observation_space.n  # perform one hot encoding on observations

        else:
            raise NotImplementedError(f"the icm wrapper does not support observation spaces of "
                                      f"type `{type(self.venv.observation_space)}` so far.")

        # memory / history
        self.prev_observations = self._process_obs_if_needed(venv.reset())
        self.o_shape = self.prev_observations[0].shape
        self.actions = None
        # memory / history - params
        self.memory_capacity = memory_capacity
        self.shuffle_memory_samples = shuffle_memory_samples
        self.postprocess_sample_size = postprocess_sample_size
        self.clear_memory_on_reset = clear_memory_on_reset
        self.clear_memory_after_n_episodes = clear_memory_after_n_episodes

        # memory / history

        self.prev_obs_memory = deque(maxlen=self.memory_capacity)
        self.obs_memory = deque(maxlen=self.memory_capacity)
        self.action_memory = deque(maxlen=self.memory_capacity)


        # parameters
        self.feature_dim = feature_dim
        self.action_dim = venv.action_space.n

        # hyper parameters
        self.beta = beta  # β
        self.eta = eta  # η
        self.lr = lr

        # neural networks
        # ᵖʳᵉᵈ means predicted or 'hat' in paper notation

        # s ⟼ Φ
        self._curiosity_feature_net = _create_fc_net(
            [self._observation_dim] + list(feature_net_hiddens) + [self.feature_dim],
            "relu",
            name="inverse_net",
        )

        # Φ(sₜ), Φ(sₜ₊₁) ⟼ aₜᵖʳᵉᵈ
        self._curiosity_inverse_fcnet = _create_fc_net(
            [2 * self.feature_dim] + list(inverse_feature_net_hiddens) + [self.action_dim],
            "relu",
            name="inverse_net",
        )

        # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
        self._curiosity_forward_fcnet = _create_fc_net(
            [self.feature_dim + self.action_dim] + list(forward_fcnet_net_hiddens) + [self.feature_dim],
            "relu",
            name="forward_net",
        )

        # see ray impl 'get_exploration_optimizer'
        feature_params = list(self._curiosity_feature_net.parameters())
        inverse_params = list(self._curiosity_inverse_fcnet.parameters())
        forward_params = list(self._curiosity_forward_fcnet.parameters())

        self._curiosity_feature_net = self._curiosity_feature_net.to(self.device)
        self._curiosity_inverse_fcnet = self._curiosity_inverse_fcnet.to(self.device)
        self._curiosity_forward_fcnet = self._curiosity_forward_fcnet.to(self.device)

        self._optimizer = torch.optim.Adam(forward_params + inverse_params + feature_params, lr=self.lr)

        # statistics
        # for intrinsic_return
        self._intrinsic_rewards = [np.array([]) for _ in range(self.venv.num_envs)]
        # for extrinsic_return
        self._extrinsic_rewards = [np.array([]) for _ in range(self.venv.num_envs)]
        #
        self.n_postprocessings = 0

    def _one_hot_encoding(self, venv_observations: np.ndarray, n: int = None) -> np.ndarray:
        if not n:
            if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                n = self.venv.observation_space.n
            else:
                raise RuntimeError("specify an `n` for `_one_hot_encoding`")
        one_hot_enc = lambda obs: np.eye(n)[obs]
        return one_hot_enc(venv_observations)

    def _process_obs_if_needed(self, venv_observations: np.ndarray):
        if isinstance(self.venv.observation_space, gym.spaces.Box):
            return venv_observations
        elif isinstance(self.venv.observation_space, gym.spaces.Discrete):
            v_obs = copy(venv_observations)
            return self._one_hot_encoding(v_obs)
        else:
            raise NotImplementedError(f"the icm wrapper does not support observation spaces of "
                                      f"type `{type(self.venv.observation_space)}` so far.")

    def _add_info_field_with_default_values(self, rewards: np.ndarray, dones: np.ndarray, infos: List[Dict]) \
            -> List[Dict]:

        for i, done in enumerate(dones):
            infos[i]["extrinsic_reward"] = rewards[i]
            infos[i]['intrinsic_reward'] = 0.0  # will be eventually overwritten later on
            infos[i]['bonus_reward'] = 0.0  # same as 'intrinsic_reward'

            if done:
                infos[i]["extrinsic_return"] = self._extrinsic_rewards[i].sum()
                infos[i]['intrinsic_return'] = self._intrinsic_rewards[i].sum()
                infos[i]['bonus_return'] = self._intrinsic_rewards[i].sum()
                infos[i]['total_return'] = infos[i]["extrinsic_return"] + infos[i]['intrinsic_return']

        return infos

    def _rest_statistics_on_episode_end(self, dones: np.ndarray):
        for i, done in enumerate(dones):
            if done:
                self._extrinsic_rewards[i] = np.array([])
                self._intrinsic_rewards[i] = np.array([])

    def _rest_icm_memory(self):
        self.prev_obs_memory = deque(maxlen=self.memory_capacity)
        self.obs_memory = deque(maxlen=self.memory_capacity)
        self.action_memory = deque(maxlen=self.memory_capacity)

    def step_async(self, actions: np.ndarray) -> None:
        self._num_timesteps += self.venv.num_envs  # one step per env
        self._num_timesteps_since_last_postprocessing += self.venv.num_envs  # one step per env
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()
        # return orignal obs at the end
        original_obs = observations
        # use processed obs for icm
        observations = self._process_obs_if_needed(observations)
        intrinsic_rewards = np.zeros(self.venv.num_envs)
        # extend info fields with default values (`extrinsic_reward`, `intrinsic_reward`, etc.)
        infos = self._add_info_field_with_default_values(rewards=rewards, dones=dones, infos=infos)

        # return if all exploration steps are done
        if self.exploration_steps is not None and self._num_timesteps > self.exploration_steps:
            self._rest_statistics_on_episode_end(dones=dones)
            return original_obs, rewards, dones, infos

        # add obs to memory and calc intrinsic reward
        if self.prev_observations is not None and self.actions is not None:
            # calc intrinsic reward
            intrinsic_rewards = self._calc_intrinsic_reward(
                obs=self.prev_observations,
                actions=self.actions,
                next_obs=observations
            )
            # add (prev_obs, action, obs) tuples to memory (one tuple for each env in the venv)
            self.prev_obs_memory.extend(self.prev_observations)
            self.action_memory.extend(self.actions)
            self.obs_memory.extend(observations)

        # add extrinsic rewards to intrinsic rewards
        augmented_reward = rewards + intrinsic_rewards

        # trigger for postprocess_trajectory
        if self._num_timesteps_since_last_postprocessing >= self.postprocess_every_n_steps:
            infos = self.postprocess_trajectory(infos=infos)
            self._num_timesteps_since_last_postprocessing = 0

        # increment episode counter
        self._num_episodes += dones.sum()
        self._num_episodes_since_cleared_icm_memory += dones.sum()

        # trigger for clearing memory
        if self.clear_memory_after_n_episodes \
                and self._num_episodes_since_cleared_icm_memory >= self.clear_memory_after_n_episodes:
            self._rest_icm_memory()

        # override previous observations
        self.prev_observations = observations
        # rest stats if it is the end of an episode
        self._rest_statistics_on_episode_end(dones=dones)

        return original_obs, augmented_reward, dones, infos

    def _calc_intrinsic_reward(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray):
        # print("_calc_intrinsic_reward")
        with torch.no_grad():
            # Push both observations through feature net to get both phis.
            # sₜ ≙ obs
            # sₜ₊₁ ≙ next_obs

            # sₜ ⟼ Φ(sₜ)
            # sₜ₊₁ ⟼ Φ(sₜ₊₁)
            phis = self._curiosity_feature_net(
                torch.cat([
                    # obs
                    torch.from_numpy(obs.astype(np.float32)).to(
                        self.device
                    ),
                    # next obs
                    torch.from_numpy(next_obs.astype(np.float32)).to(
                        self.device
                    )
                ])
            )
            # Φ(sₜ), Φ(sₜ₊₁)
            phi, next_phi = torch.chunk(phis, 2)

            # Predict next phi with forward model.
            # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
            #
            # Equation 4
            #
            # Φ(sₜ₊₁)ᵖʳᵉᵈ  =  f(Φ(sₜ),aₜ;θᵢ)
            actions_tensor = (
                torch.from_numpy(actions).long().to(self.device)
            )
            predicted_next_phi = self._curiosity_forward_fcnet(
                # note: ray uses an adapter for one_hot that also supports MultiDiscrete action spaces
                # using the native torch one_hot function here to avoid dependencies on ray/rllib
                torch.cat([
                    phi,
                    one_hot(actions_tensor, self.action_space.n).float()
                ],
                    dim=-1)
            )
            # print(f"{predicted_next_phi.shape}, {predicted_next_phi=}")

            # Forward loss term (predicted phi', given phi and action vs actually
            # observed phi').
            #
            #  Equation 5
            #
            #     ⎛                  ⎞     1                           2
            #  Lբ ⎜ Φ(sₜ), Φ(sₜ₊₁)ᵖʳᵉᵈ⎟ =  ───  ║ Φ(sₜ₊₁)ᵖʳᵉᵈ - Φ(sₜ)  ║₂
            #     ⎝                  ⎠     2
            #
            forward_l2_norm_sqared = 0.5 * torch.sum(
               torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
            )

            intrinsic_rewards = self.eta * forward_l2_norm_sqared.detach().cpu().numpy()

        return intrinsic_rewards

    def postprocess_trajectory(self, infos: List[Dict]) -> List[Dict]:
        self.n_postprocessings += 1
        # print("post")

        obs = np.array(self.prev_obs_memory)
        actions = np.array(self.action_memory)
        next_obs = np.array(self.obs_memory)

        # Push both observations through feature net to get both phis.
        # sₜ ≙ obs
        # sₜ₊₁ ≙ next_obs

        # sₜ ⟼ Φ(sₜ)
        # sₜ₊₁ ⟼ Φ(sₜ₊₁)
        phis = self._curiosity_feature_net(
            torch.cat([
                # obs
                torch.from_numpy(obs.astype(np.float32)).to(
                    self.device
                ),
                # next obs
                torch.from_numpy(next_obs.astype(np.float32)).to(
                    self.device
                )
            ])
        )
        # Φ(sₜ), Φ(sₜ₊₁)
        phi, next_phi = torch.chunk(phis, 2)

        # Predict next phi with forward model.
        # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
        #
        # Equation 4
        #
        # Φ(sₜ₊₁)ᵖʳᵉᵈ  =  f(Φ(sₜ),aₜ;θᵢ)
        actions_tensor = (
            torch.from_numpy(actions).long().to(self.device)
        )
        predicted_next_phi = self._curiosity_forward_fcnet(
            # note: ray uses an adapter for one_hot that also supports MultiDiscrete action spaces
            # using the native torch one_hot function here to avoid dependencies on ray/rllib
            torch.cat([
                phi,
                one_hot(actions_tensor, self.action_space.n).float()
            ],
                dim=-1)
        )
        # print(f"{predicted_next_phi.shape}, {predicted_next_phi=}")

        # Forward loss term (predicted phi', given phi and action vs actually
        # observed phi').
        #
        #  Equation 5
        #
        #     ⎛                  ⎞     1                           2
        #  Lբ ⎜ Φ(sₜ), Φ(sₜ₊₁)ᵖʳᵉᵈ⎟ =  ───  ║ Φ(sₜ₊₁)ᵖʳᵉᵈ - Φ(sₜ)  ║₂
        #     ⎝                  ⎠     2
        #
        forward_l2_norm_sqared = 0.5 * torch.sum(
           torch.pow(predicted_next_phi - next_phi, 2.0), dim=-1
        )
        forward_loss = torch.mean(forward_l2_norm_sqared)
        # forward_loss = nn.MSELoss(predicted_next_phi, next_phi)
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
        # ToDo: check section
        # print(f"{self._curiosity_inverse_fcnet}")
        dist_inputs = self._curiosity_inverse_fcnet(phi_cat_next_phi)
        # print(f"{dist_inputs.shape} {dist_inputs=}")
        # from torch.distributions.utils import logits_to_probs
        # probs = logits_to_probs(dist_inputs)
        # print(f"{probs.shape} {probs=}")
        # action_dist = torch.distributions.categorical.Categorical(probs=probs)
        action_dist = torch.distributions.categorical.Categorical(logits=dist_inputs)
        # print(f"{action_dist=}")
        # Neg log(p); p=probability of observed action given the inverse-NN
        # predicted action distribution.
        # print(f"{actions_tensor.shape} {actions_tensor=}")
        inverse_loss = -action_dist.log_prob(actions_tensor)
        # print(f"{inverse_loss.shape} {inverse_loss=}")
        inverse_loss = torch.mean(inverse_loss)
        # print(f"mean {inverse_loss.shape} {inverse_loss=}")

        # Calculate the ICM loss.
        loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        print(
            f"loss: {loss.item():.4f}, forward_loss: {forward_loss.item():.4f}, inverse_loss: {inverse_loss.item():.4f}")

        # Perform an optimizer step.
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # add infos to info dict
        for info in infos:
            info["n_postprocessings"] = self.n_postprocessings
            info["loss"] = loss.item()
            info["inverse_loss"] = inverse_loss.item()
            info["forward_loss"] = forward_loss.item()

        return infos

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()
        original_obs = observations

        if self.clear_memory_on_reset:
            self._rest_icm_memory()

        return original_obs


if __name__ == '__main__':
    from gym.wrappers import TimeLimit
    from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3 import A2C, PPO

    print("##### CartPole-v1 #####")
    budget = 1_000
    eval_episodes = 10
    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=4
    )
    cartpole_venv = VecMonitor(venv=venv)
    # model1 = A2C('MlpPolicy', cartpole_venv, verbose=0, seed=773)
    # model1.learn(total_timesteps=budget)
    # mean_reward, std_reward = evaluate_policy(model1, cartpole_venv, n_eval_episodes=eval_episodes)
    # print(f"without icm: {mean_reward=}, {std_reward=}")
    cartpole_venv.reset()
    cartpole_icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
        eta=0.15,
        exploration_steps=int(budget * 0.5)
    )
    cartpole_icm_venv = VecMonitor(venv=cartpole_icm_venv)
    model2 = A2C('MlpPolicy', cartpole_icm_venv, verbose=0, seed=773)
    # model2.set_env(cartpole_venv)
    # model2.learn(total_timesteps=budget)
    # mean_reward, std_reward = evaluate_policy(model2, model2.get_env(), n_eval_episodes=eval_episodes)
    # print(f"with icm: {mean_reward=}, {std_reward=}")

    print("#### FrozenLake-v1 #####")

    budget = 10_000
    eval_episodes = 10

    env_name = "FrozenLake-v1"
    env_kwargs = {
        "desc": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFG",
        ],
        "is_slippery": False,
    }

    venv = make_vec_env_without_monitor(
        env_id=env_name,
        env_kwargs=env_kwargs,
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": 20},
        n_envs=1
    )

    # no_icm_venv = VecMonitor(venv=venv)
    # no_icm_model = PPO('MlpPolicy', no_icm_venv, verbose=0)
    # no_icm_model.learn(total_timesteps=budget)

    icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
        postprocess_every_n_steps=1 * 16,
        exploration_steps=int(0.5 * budget),
        feature_net_hiddens=[],
        forward_fcnet_net_hiddens=[256],
        inverse_feature_net_hiddens=[256],
        memory_capacity=16,
        postprocess_sample_size=16,

    )
    icm_venv = VecMonitor(venv=icm_venv)
    icm_model = PPO('MlpPolicy', icm_venv, verbose=0)
    icm_model.learn(total_timesteps=budget)

    mean_reward, std_reward = evaluate_policy(icm_model, icm_model.get_env(), n_eval_episodes=eval_episodes)
    print(f"with icm: {mean_reward=}, {std_reward=}")

    print("#### MiniGrid-Empty-5x5-v0 #### ")

    budget = 10_000
    eval_episodes = 10

    import gym_minigrid
    from jss_rl.sb3.util.minigrid_one_hot_wrapper import OneHotWrapper


    def make_env():
        env = gym.make("MiniGrid-Empty-5x5-v0")
        env = gym_minigrid.wrappers.ImgObsWrapper(env)
        framestack = 4
        env = OneHotWrapper(
            env,
            0,
            framestack=framestack,
        )
        return env


    venv = DummyVecEnv([lambda: make_env()])

    icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
        exploration_steps=int(0.5 * budget),
        postprocess_every_n_steps=4 * 15
    )
    icm_venv = VecMonitor(
        venv=icm_venv
    )
    icm_model = PPO('MlpPolicy', icm_venv, verbose=0)
    icm_model.learn(total_timesteps=budget)

    # mean_reward, std_reward = evaluate_policy(icm_model, icm_model.get_env(), n_eval_episodes=eval_episodes)
    # print(f"with icm: {mean_reward=}, {std_reward=}")
