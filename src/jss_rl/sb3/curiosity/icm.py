import itertools
from collections import deque
from copy import copy

import gym
import torch

from typing import Union, List, Dict

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnv, VecEnvObs
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv

from jss_rl.sb3.moving_avarage import MovingAverage
from jss_rl.sb3.torch_dense_sequential_model_builder import _create_fc_net
from torch.nn.functional import one_hot


class IntrinsicCuriosityModuleWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: Union[VecEnvWrapper, VecEnv, DummyVecEnv],

                 beta: float = 0.2,
                 eta: float = 1.0,
                 lr: float = 1e-3,

                 device: str = 'cpu',

                 feature_dim: int = 288,
                 feature_net_hiddens: List[int] = None,
                 feature_net_activation: str = "relu",
                 inverse_feature_net_hiddens: List[int] = None,
                 inverse_feature_net_activation: str = "relu",
                 forward_fcnet_net_hiddens: List[int] = None,
                 forward_fcnet_net_activation: str = "relu",

                 memory_capacity: int = 100,
                 clear_memory_on_end_of_episode: bool = False,
                 clear_memory_every_n_steps: int = None,
                 shuffle_samples: bool = True,
                 maximum_sample_size: int = None,
                 postprocess_on_end_of_episode: bool = True,
                 postprocess_every_n_steps: int = None,

                 exploration_steps: int = None,
                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        self._num_timesteps = 0

        self.device = device

        self.actions = None
        self.prev_observations = self._process_obs_if_needed(venv.reset())

        # memory / history - instance
        # three deques for every subenv
        self.memory_capacity = memory_capacity
        self.prev_obs_memory = [deque(maxlen=memory_capacity) for _ in range(self.venv.num_envs)]
        self.obs_memory = [deque(maxlen=memory_capacity) for _ in range(self.venv.num_envs)]
        self.action_memory = [deque(maxlen=memory_capacity) for _ in range(self.venv.num_envs)]

        self.clear_memory_on_end_of_episode = clear_memory_on_end_of_episode
        self.clear_memory_every_n_steps = clear_memory_every_n_steps

        self.postprocess_on_end_of_episode = postprocess_on_end_of_episode
        self.postprocess_every_n_steps = postprocess_every_n_steps


        self.shuffle_samples = shuffle_samples
        self.maximum_sample_size = maximum_sample_size

        self.exploration_steps = exploration_steps

        self._clear_memory_counter = 0
        self._postprocess_counter = 0

        # hyper parameters
        self.beta = beta  # β
        self.eta = eta  # η
        self.lr = lr

        if feature_net_hiddens is None:
            feature_net_hiddens = [256]

        if inverse_feature_net_hiddens is None:
            inverse_feature_net_hiddens = [256]

        if forward_fcnet_net_hiddens is None:
            forward_fcnet_net_hiddens = [256]

        self.feature_net_activation = feature_net_activation
        self.inverse_feature_net_activation = inverse_feature_net_activation
        self.forward_fcnet_net_activation = forward_fcnet_net_activation

        self.feature_dim = feature_dim
        self._observation_dim = self.prev_observations[0].ravel().shape[0]

        if not isinstance(venv.action_space, gym.spaces.Discrete):
            raise NotImplementedError

        self.action_dim = venv.action_space.n

        # neural networks
        # ᵖʳᵉᵈ means predicted or 'hat' in paper notation

        # s ⟼ Φ
        self._curiosity_feature_net = _create_fc_net(
            [self._observation_dim] + list(feature_net_hiddens) + [self.feature_dim],
            self.feature_net_activation,
            name="inverse_net",
        )

        # Φ(sₜ), Φ(sₜ₊₁) ⟼ aₜᵖʳᵉᵈ
        self._curiosity_inverse_fcnet = _create_fc_net(
            [2 * self.feature_dim] + list(inverse_feature_net_hiddens) + [self.action_dim],
            self.inverse_feature_net_activation,
            name="inverse_net",
        )

        # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
        self._curiosity_forward_fcnet = _create_fc_net(
            [self.feature_dim + self.action_dim] + list(forward_fcnet_net_hiddens) + [self.feature_dim],
            self.forward_fcnet_net_activation,
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
        self.n_postprocessings = 0
        self.global_stats = {
            "n_total_episodes": 0,
            "n_postprocessings": self.n_postprocessings,
            "memory_size": len(self.obs_memory)
        }
        self.sub_env_stats = [{
            "extrinsic_rewards": [],
            "intrinsic_rewards": [],
            "n_sub_env_episodes": 0,
        } for _ in range(self.venv.num_envs)]

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
            # flatten not flat spaces
            return np.array([venv_observations[i].ravel() for i in range(self.venv.num_envs)])
        elif isinstance(self.venv.observation_space, gym.spaces.Discrete):
            # one hot encoding for discrete obs spaces
            v_obs = copy(venv_observations)
            return self._one_hot_encoding(v_obs)
        else:
            raise NotImplementedError(f"the icm wrapper does not support observation spaces of "
                                      f"type `{type(self.venv.observation_space)}` so far.")

    def step_async(self, actions: np.ndarray) -> None:
        self._num_timesteps += self.venv.num_envs  # one step per env
        self._postprocess_counter += self.venv.num_envs
        self._clear_memory_counter += self.venv.num_envs
        self.actions = actions
        self.venv.step_async(actions)

    def _extend_infos(self, augmented_rewards: np.ndarray, original_rewards: np.ndarray, intrinsic_rewards: np.ndarray,
                      dones: np.ndarray, infos) -> List[Dict]:

        self.global_stats["n_total_episodes"] += dones.sum()
        self.global_stats["n_postprocessings"] = self.n_postprocessings
        self.global_stats["_num_timesteps"] = self._num_timesteps
        self.global_stats["memory_size"] = len(self.obs_memory)

        extended_infos = [info.copy() for info in infos]

        for i in range(self.venv.num_envs):
            self.sub_env_stats[i]["extrinsic_rewards"].append(original_rewards[i])
            self.sub_env_stats[i]["intrinsic_rewards"].append(intrinsic_rewards[i])

            if dones[i]:
                self.sub_env_stats[i]["n_sub_env_episodes"] += 1

                extended_infos[i]["extrinsic_return"] = sum(self.sub_env_stats[i]["extrinsic_rewards"])
                self.sub_env_stats[i]["extrinsic_rewards"] = []

                extended_infos[i]["intrinsic_return"] = sum(self.sub_env_stats[i]["intrinsic_rewards"])
                self.sub_env_stats[i]["intrinsic_rewards"] = []

                extended_infos[i]["bonus_return"] = extended_infos[i]["intrinsic_return"]
                extended_infos[i]["total_return"] = \
                    extended_infos[i]["intrinsic_return"] + extended_infos[i]["extrinsic_return"]

            extended_infos[i]["extrinsic_reward"] = original_rewards[i]
            extended_infos[i]["intrinsic_reward"] = intrinsic_rewards[i]
            extended_infos[i]["bonus_reward"] = intrinsic_rewards[i]
            extended_infos[i]["total_reward"] = augmented_rewards[i]
            extended_infos[i]["n_total_episodes"] = self.global_stats["n_total_episodes"]
            extended_infos[i]["n_postprocessings"] = self.global_stats["n_postprocessings"]
            extended_infos[i]["_num_timesteps"] = self.global_stats["_num_timesteps"]
            extended_infos[i]["memory_size"] = self.global_stats["memory_size"]


        return extended_infos

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()
        # return original obs at the end
        original_obs = observations
        observations = self._process_obs_if_needed(observations)

        intrinsic_rewards = np.zeros(self.venv.num_envs)

        # return if all exploration steps are done
        if self.exploration_steps is not None and self._num_timesteps > self.exploration_steps:
            # self._rest_statistics_on_episode_end(dones=dones)
            extended_infos = self._extend_infos(
                augmented_rewards=rewards,
                original_rewards=rewards,
                intrinsic_rewards=intrinsic_rewards,
                dones=dones,
                infos=infos,
            )
            return original_obs, rewards, dones, extended_infos

        # add tuples to memory
        for i in range(self.venv.num_envs):
            # prev observations
            self.prev_obs_memory[i].append(self.prev_observations[i])
            self.action_memory[i].append(self.actions[i])
            self.obs_memory[i].append(observations[i])

        # check for episode end
        for i, done in enumerate(dones):
            if done:
                # postprocess
                if self.postprocess_on_end_of_episode:
                    post_process_infos = self.postprocess_trajectory(
                        obs=self.prev_obs_memory[i],
                        next_obs=self.obs_memory[i],
                        actions=self.action_memory[i]
                    )
                    # add infos to infos dicts
                    infos[i] = {**post_process_infos, **infos[i]}
                # reset memory
                if self.clear_memory_on_end_of_episode:
                    self.prev_obs_memory[i] = deque(maxlen=self.memory_capacity)
                    self.action_memory[i] = deque(maxlen=self.memory_capacity)
                    self.obs_memory[i] = deque(maxlen=self.memory_capacity)

        # trigger for postprocessing
        if self.postprocess_every_n_steps and self._postprocess_counter >= self.postprocess_every_n_steps:

            # concat values from all sub envs
            prev_obs = list(itertools.chain.from_iterable(self.prev_obs_memory))
            actions = list(itertools.chain.from_iterable(self.action_memory))
            obs = list(itertools.chain.from_iterable(self.obs_memory))

            post_process_infos = self.postprocess_trajectory(
                obs=prev_obs,
                actions=actions,
                next_obs=obs
            )
            # add infos to infos dicts
            for i in range(self.venv.num_envs):
                infos[i] = {**post_process_infos, **infos[i]}

            self._postprocess_counter = 0

        # add obs to memory and calc intrinsic reward
        if self.prev_observations is not None and self.actions is not None:
            # calc intrinsic reward
            intrinsic_rewards = self._calc_intrinsic_reward(
                obs=self.prev_observations,
                actions=self.actions,
                next_obs=observations
            )

        augmented_rewards = rewards + intrinsic_rewards

        # trigger for memory reset
        if self.clear_memory_every_n_steps and self._clear_memory_counter >= self.clear_memory_every_n_steps:
            self.prev_obs_memory = [deque(maxlen=self.memory_capacity) for _ in range(self.venv.num_envs)]
            self.obs_memory = [deque(maxlen=self.memory_capacity) for _ in range(self.venv.num_envs)]
            self.action_memory = [deque(maxlen=self.memory_capacity) for _ in range(self.venv.num_envs)]

            self._clear_memory_counter = 0

        # override previous observations
        self.prev_observations = observations


        extended_infos = self._extend_infos(
            augmented_rewards=augmented_rewards,
            original_rewards=rewards,
            intrinsic_rewards=intrinsic_rewards,
            dones=dones,
            infos=infos,
        )

        return original_obs, augmented_rewards, dones, extended_infos

    def postprocess_trajectory(self, obs, next_obs, actions) -> Dict[str, float]:

        self.n_postprocessings += 1

        if self.shuffle_samples:
            from sklearn.utils import shuffle
            obs, actions, next_obs = shuffle(obs, actions, next_obs)

        if self.maximum_sample_size:
            obs = obs[:self.maximum_sample_size]
            actions = actions[:self.maximum_sample_size]
            next_obs = next_obs[:self.maximum_sample_size]

        # print(f"obs: {obs}, len next obs: {next_obs}, actions: {actions}")
        # print(f"len obs: {len(obs)}, len next obs: {len(next_obs)}, actions: {len(actions)}")

        # Push both observations through feature net to get both phis.
        # sₜ ≙ obs
        # sₜ₊₁ ≙ next_obs

        # sₜ ⟼ Φ(sₜ)
        # sₜ₊₁ ⟼ Φ(sₜ₊₁)
        phis = self._curiosity_feature_net(
            torch.cat([
                # obs
                torch.from_numpy(np.array(obs).astype(np.float32)).to(
                    self.device
                ),
                # next obs
                torch.from_numpy(np.array(next_obs).astype(np.float32)).to(
                    self.device
                )
            ])
        )
        # print(f"phis: {phis.shape}")
        phi, next_phi = torch.chunk(phis, 2)
        # print(f"phi: {phi.shape}, next_phi: {next_phi.shape}")

        # Predict next phi with forward model.
        # Φ(sₜ), aₜ ⟼ Φ(sₜ₊₁)ᵖʳᵉᵈ
        #
        # Equation 4
        #
        # Φ(sₜ₊₁)ᵖʳᵉᵈ  =  f(Φ(sₜ),aₜ;θᵢ)
        actions_tensor = (
            torch.from_numpy(
                np.array(actions).astype(np.float32)
            ).to(self.device).long().to(self.device)
        )
        # print(f"actions_tensor: {actions_tensor.shape}")
        predicted_next_phi = self._curiosity_forward_fcnet(
            # note: ray uses an adapter for one_hot that also supports MultiDiscrete action spaces
            # using the native torch one_hot function here to avoid dependencies on ray/rllib
            torch.cat([
                phi,
                one_hot(actions_tensor, self.action_space.n).float()
            ],
                dim=-1)
        )
        # print(f"predicted_next_phi: {predicted_next_phi.shape}")

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
        # print(f"forward_l2_norm_sqared: {forward_l2_norm_sqared}")
        forward_loss = torch.mean(forward_l2_norm_sqared)
        # print(f"forward_loss: {forward_loss}")

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
            "loss": loss.item(),
            "inverse_loss": inverse_loss.item(),
            "forward_loss": forward_loss.item(),
        }

    def _calc_intrinsic_reward(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray):
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

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        print("reset")

        observations = self.venv.reset()
        original_obs = observations

        # override previous observations
        self.prev_observations = self._process_obs_if_needed(observations)

        # reset all memories
        self.prev_obs_memory = [deque(maxlen=self.memory_capacity) for _ in range(self.venv.num_envs)]
        self.obs_memory = [deque(maxlen=self.memory_capacity) for _ in range(self.venv.num_envs)]
        self.action_memory = [deque(maxlen=self.memory_capacity) for _ in range(self.venv.num_envs)]

        return original_obs


if __name__ == '__main__':
    from gym.wrappers import TimeLimit
    from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
    from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper, DummyVecEnv
    from stable_baselines3 import PPO

    print("##### CartPole-v1 #####")
    budget = 10_000
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
    )

    icm_model = PPO('MlpPolicy', cartpole_icm_venv, verbose=0)
    icm_model.learn(total_timesteps=budget)

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
        wrapper_kwargs={"max_episode_steps": 16},
        n_envs=1
    )


    class DistanceWrapper(VecEnvWrapper):

        def __init__(self, venv):
            self.distances = MovingAverage(capacity=1000)
            VecEnvWrapper.__init__(self, venv=venv)
            self._steps = 0

        def step_wait(self) -> VecEnvStepReturn:
            """Overrides VecEnvWrapper.step_wait."""
            observations, rewards, dones, infos = self.venv.step_wait()
            self._steps += self.venv.num_envs

            for i, o in enumerate(observations):
                x, y = o % 8, o // 8  # frozen lake with 8x8 size
                distance_from_origin = (x ** 2 + y ** 2) ** 0.5
                self.distances.add(distance_from_origin)
                print(f"[{self._steps}] distance_from_origin: {distance_from_origin:.4f},"
                      f" moving avarage distance_from_origin: {self.distances.mean():.4f}")

            return observations, rewards, dones, infos

        def reset(self) -> VecEnvObs:
            """Overrides VecEnvWrapper.reset."""
            observations = self.venv.reset()
            return observations


    venv = DistanceWrapper(venv=venv)

    # no_icm_venv = VecMonitor(venv=venv)
    # no_icm_model = PPO('MlpPolicy', no_icm_venv, verbose=0)
    # no_icm_model.learn(total_timesteps=budget)

    icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
        exploration_steps=int(0.5 * budget),
        feature_net_hiddens=[],
        forward_fcnet_net_hiddens=[256],
        inverse_feature_net_hiddens=[256],

        maximum_sample_size=16,

        clear_memory_on_end_of_episode=True,
        postprocess_on_end_of_episode=True,

        clear_memory_every_n_steps=None,
        postprocess_every_n_steps=None,
    )
    icm_venv = VecMonitor(venv=icm_venv)
    icm_model = PPO('MlpPolicy', icm_venv, verbose=0)
    icm_model.learn(total_timesteps=budget)

    # mean_reward, std_reward = evaluate_policy(icm_model, icm_model.get_env(), n_eval_episodes=eval_episodes)
    # print(f"with icm: {mean_reward=}, {std_reward=}")
