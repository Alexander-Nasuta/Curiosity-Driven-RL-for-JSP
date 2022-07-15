import itertools
import sys
from collections import deque
from copy import copy

import gym
import torch
import torch.nn as nn

from typing import Union, List, Dict, Callable

import numpy as np
from ray.rllib.models.torch.misc import SlimFC
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvWrapper, VecEnv, VecEnvObs
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv

from jss_rl.sb3.util.moving_avarage import MovingAverage
from jss_rl.sb3.util.torch_dense_sequential_model_builder import _create_fc_net
from torch.nn.functional import one_hot


class EpisodicCuriosityModuleWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: Union[VecEnvWrapper, VecEnv, DummyVecEnv],

                 embedding_dim: int = 288,
                 embedding_net_hiddens: List[int] = None,
                 embedding_net_activation: str = "relu",
                 comparator_net_hiddens: List[int] = None,
                 comparator_net_activation: str = "relu",
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 lr: float = 1e-3,
                 k: int = 2,
                 gamma: int = 3,
                 b_novelty: float = 0.0,
                 episodic_memory_capacity: int = 100,
                 clear_memory_every_episode: bool = True,
                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        # default params
        if embedding_net_hiddens is None:
            embedding_net_hiddens = [256]

        if comparator_net_hiddens is None:
            comparator_net_hiddens = [256]

        self._num_timesteps = 0
        self._observation_dim = self._process_obs_if_needed(self.venv.reset())[0].ravel().shape[0]

        if not isinstance(venv.action_space, gym.spaces.Discrete):
            raise NotImplementedError

        self.action_dim = venv.action_space.n

        self.embedding_dim = embedding_dim
        self.embedding_net_hiddens = embedding_net_hiddens
        self.embedding_net_activation = embedding_net_activation

        self.comparator_net_hiddens = comparator_net_hiddens
        self.comparator_net_activation = comparator_net_activation

        # The module consists of both parametric and non-parametric components.

        # There are two non-parametric components:
        #   episodic memory buffer 𝑴
        #   reward bonus estimation function 𝑩

        # memory buffer 𝑴
        self.episodic_memory_capacity = episodic_memory_capacity
        #
        self.ec_memory_buffers = [deque(maxlen=self.episodic_memory_capacity) for _ in range(self.venv.num_envs)]

        initial_obs = self._process_obs_if_needed(self.venv.reset())

        # reward bonus estimation function 𝑩
        # α ∈ R+
        # β = 0.5 works well for fixed-duration episodes,
        # and β = 1 is preferred if an episode could have variable length.

        # hyper-param for training
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.k = k
        self.gamma = gamma
        self.b_novelty = b_novelty
        self.clear_memory_every_episode = clear_memory_every_episode

        # bₙₒᵥₑₗₜᵧ
        # `After the bonus computation, the observation embedding is added to memory if the bonus b is larger
        #  than a novelty threshold bₙₒᵥₑₗₜᵧ`

        # There are two parametric components:
        #   embedding network 𝑬 : 𝒪 → ℝⁿ
        #   comparator network 𝑪 : ℝⁿ × ℝⁿ → [0, 1].

        # embedding network 𝑬 : 𝒪 → ℝⁿ
        self._embedding_net = _create_fc_net(
            [self._observation_dim] + list(self.embedding_net_hiddens) + [self.embedding_dim],
            self.embedding_net_activation,
            name="embedding_net",
        )

        # comparator network 𝑪 : ℝⁿ × ℝⁿ → [0, 1].
        self._comparator_net = _create_fc_net(
            [2 * self.embedding_dim] + list(self.comparator_net_hiddens) + [1],
            self.comparator_net_activation,
            name="comparator_net",
        )

        embedding_params = list(self._embedding_net.parameters())
        comparator_params = list(self._comparator_net.parameters())

        self._optimizer = torch.optim.Adam(embedding_params + comparator_params, lr=self.lr)

        # one trajectory per env
        # append new obs on every step
        self.trajectories = [
            [initial_obs[i]] for i in range(self.venv.num_envs)
        ]

        # statistics
        self.n_postprocessings = 0
        self.global_stats = {
            "n_total_episodes": 0,
            "n_postprocessings": self.n_postprocessings,
            "memory_size": 0
        }
        self.sub_env_stats = [{
            "extrinsic_rewards": [],
            "intrinsic_rewards": [],
            "n_sub_env_episodes": 0,
        } for _ in range(self.venv.num_envs)]

    def reward_bonus_funktion(self, similarity_score: float):
        # b = B(M, e) = α(β − C(M, e))
        return self.alpha * (self.beta - similarity_score)

    def _calc_reward_bonus(self, env_i: int, single_observation: np.ndarray) -> (float, Dict):
        with torch.no_grad():
            single_embedded_observation = self._embedding_net(
                torch.from_numpy(np.array(single_observation).astype(np.float32))
            )

            if not len(self.ec_memory_buffers[env_i]):
                self.ec_memory_buffers[env_i].append(single_embedded_observation)
                return 0.0, {
                    "_bonus": 0.0,
                }

            # pair single_embedded_observation with all entries in episodic memory
            pairs = torch.empty(size=(0, self.embedding_dim * 2), dtype=torch.float32)

            for memory_entry in self.ec_memory_buffers[env_i]:
                pair = torch.cat([memory_entry, single_embedded_observation])
                pair = torch.reshape(pair, (1, self.embedding_dim * 2))
                pairs = torch.cat((pairs, pair), 0)

            # print(f"pairs: {pairs}")

            reachability_buffer = self._comparator_net(pairs).cpu().detach().numpy()
            # print(f"reachability_buffer: {reachability_buffer}")

            # aggregation
            # the aggregation function 𝑭 then maps the reachability buffer to [0.0, 1.0]
            percentile = 90
            similarity_score = np.percentile(reachability_buffer, percentile)
            # print(f"similarity_score: {similarity_score}")

            # calc bonus b
            b = self.reward_bonus_funktion(similarity_score=similarity_score)
            # print(f"b: {b}")

            # `After the bonus computation, the observation embedding is added to memory if the bonus b is
            #  larger than a novelty threshold bₙₒᵥₑₗₜᵧ`
            #
            # if current observation is novel add b to boni (boni are added to the rewads after the loop)
            if b > self.b_novelty:
                self.ec_memory_buffers[env_i].append(single_embedded_observation)
                return b, {
                    "similarity_score": similarity_score,
                    "_bonus": b,
                    "novel": True,
                }
            else:
                return 0.0, {
                    "similarity_score": similarity_score,
                    "_bonus": b,
                    "novel": False,
                }

    def reset(self) -> VecEnvObs:
        observations = self.venv.reset()

        # clear memory
        for memory in self.ec_memory_buffers:
            memory.clear()

        return observations

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()
        # return original obs at the end
        original_obs = observations
        observations = self._process_obs_if_needed(observations)

        # add observations to trajectories
        for env_id, obs in enumerate(observations):
            self.trajectories[env_id].append(obs)

        # compare current observations with the ones in memory
        reward_bonus = np.zeros(self.venv.num_envs)

        # calc boni
        for env_i, single_obs in enumerate(observations):
            bonus, info = self._calc_reward_bonus(
               env_i=env_i,
               single_observation=single_obs
            )
            reward_bonus[env_i] = bonus

        # trigger code that run at the end of an episode
        for env_i, done in enumerate(dones):
            if done:
                info = self._on_episode_end(env_i=env_i)
                infos[env_i] = {**infos[env_i], **info}


        augmented_rewards = rewards + reward_bonus

        extended_infos = self._extend_infos(
            augmented_rewards=augmented_rewards,
            original_rewards=rewards,
            intrinsic_rewards=reward_bonus,
            dones=dones,
            infos=infos,
        )

        return original_obs, augmented_rewards, dones, extended_infos

    def _on_episode_end(self, env_i: int) -> Dict:
        # print(f"[env {env_i}] end of episode")
        infos = self._postprocess_trajectory(env_i=env_i)

        # clear trajectory
        self.trajectories[env_i].clear()

        # reset memory at the end
        if self.clear_memory_every_episode:
            self.ec_memory_buffers[env_i].clear()

        return infos

    def _extend_infos(self, augmented_rewards: np.ndarray, original_rewards: np.ndarray, intrinsic_rewards: np.ndarray,
                      dones: np.ndarray, infos) -> List[Dict]:

        self.global_stats["n_total_episodes"] += dones.sum()
        self.global_stats["n_postprocessings"] = self.n_postprocessings
        self.global_stats["_num_timesteps"] = self._num_timesteps
        self.global_stats["memory_size"] = sum([len(memory) for memory in self.ec_memory_buffers])

        # print(f"memory_size: {len(self.ec_memory)}")

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
            extended_infos[i]["memory_size"] = len(self.ec_memory_buffers[i])

        return extended_infos

    def _postprocess_trajectory(self, env_i: int) -> Dict:

        trajectory = np.array(self.trajectories[env_i]).astype(np.float32)
        # print(f"trajectory shape: {trajectory.shape}")

        # perform embedding on observations
        embedded_obs = self._embedding_net(torch.from_numpy(trajectory))
        # print(f"embedded_obs shape: {embedded_obs.size()}")

        if not len(self.ec_memory_buffers[env_i]):
            self.ec_memory_buffers[env_i].append(embedded_obs[0])

        pairs = torch.empty(size=(0, self.embedding_dim * 2), dtype=torch.float32)
        labels = torch.empty(size=(0, 1), dtype=torch.float32)

        # construct pairs

        # `it predicts values close to 0 if probability of two observations being reach- able from one another within
        #  k steps is low, and values close to 1 when this probability is high`

        # positive examples (close distance)
        for k in range(1, self.k + 1):
            for obs1, obs2 in zip(embedded_obs, embedded_obs[k:]):
                # x
                pair = torch.cat([obs1, obs2])
                pair = torch.reshape(pair, (1, self.embedding_dim * 2))
                pairs = torch.cat((pairs, pair), 0)
                # y
                label = torch.tensor(1)
                label = torch.reshape(label, (1, 1))
                labels = torch.cat((labels, label), 0)

        # negative examples (large distance)
        k_start = self.k * self.gamma
        k_end = k_start + self.k
        for k in range(k_start, k_end):
            for obs1, obs2 in zip(embedded_obs, embedded_obs[k:]):
                # x
                pair = torch.cat([obs1, obs2])
                pair = torch.reshape(pair, (1, self.embedding_dim * 2))
                pairs = torch.cat((pairs, pair), 0)
                # y
                label = torch.tensor(0)
                label = torch.reshape(label, (1, 1))
                labels = torch.cat((labels, label), 0)


        pred = self._comparator_net(pairs)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(pred, labels)

        # Perform an optimizer step.
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # print(f"loss: {loss.item()}")

        return {
            "loss": loss.item()
        }

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

    def _one_hot_encoding(self, venv_observations: np.ndarray, n: int = None) -> np.ndarray:
        if not n:
            if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                n = self.venv.observation_space.n
            else:
                raise RuntimeError("specify an `n` for `_one_hot_encoding`")
        one_hot_enc = lambda obs: np.eye(n)[obs]
        return one_hot_enc(venv_observations)


if __name__ == '__main__':
    from gym.wrappers import TimeLimit
    from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
    from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper, DummyVecEnv
    from stable_baselines3 import A2C, PPO

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
    cartpole_icm_venv = EpisodicCuriosityModuleWrapper(
        venv=venv,
    )

    ec_model = PPO('MlpPolicy', cartpole_icm_venv, verbose=0)
    #ec_model.learn(total_timesteps=budget)

    print("#### FrozenLake-v1 #####")

    budget = 50_000
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

    ec_venv = EpisodicCuriosityModuleWrapper(
        venv=venv,
        **{
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 2,
            "embedding_dim": 64,
            "episodic_memory_capacity": 8,
        }
    )
    ec_venv = VecMonitor(venv=ec_venv)
    ec_model = PPO('MlpPolicy', ec_venv, verbose=0)
    ec_model.learn(total_timesteps=budget)

    # mean_reward, std_reward = evaluate_policy(icm_model, icm_model.get_env(), n_eval_episodes=eval_episodes)
    # print(f"with icm: {mean_reward=}, {std_reward=}")
