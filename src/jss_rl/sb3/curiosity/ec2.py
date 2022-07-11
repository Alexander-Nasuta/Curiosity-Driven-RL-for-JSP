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

                 alpha: float = 0.1,
                 beta: float = 1.0,
                 lr: float = 1e-3,
                 gamma: int = 2,
                 novelty_threshold: float = 0.0,

                 embedding_feature_dim: int = 288,
                 embedding_net_hiddens: List[int] = None,
                 embedding_net_activation="relu",

                 comparator_net_hiddens: List[int] = None,
                 comparator_net_activation="relu",
                 similarity_aggregation='percentile',  # 'max', 'nth_largest', 'percentile', 'relative_count'
                 similarity_aggregation_percentile: int = 90,
                 similarity_aggregation_nth_largest_max: int = 10,
                 similarity_aggregation_relative_count_threshold: float = 0.5,

                 ec_memory_replacement_strategy: str = 'fifo',
                 ec_capacity: int = 6,
                 ec_memory_reset_on_episode_end: bool = True,

                 train_action_distance_threshold: int = 2,

                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        # default params
        if embedding_net_hiddens is None:
            embedding_net_hiddens = [64, 64]

        if comparator_net_hiddens is None:
            comparator_net_hiddens = [64, 64]

        self._num_timesteps = 0

        self._observation_dim = self._process_obs_if_needed(self.venv.reset())[0].ravel().shape[0]

        if not isinstance(venv.action_space, gym.spaces.Discrete):
            raise NotImplementedError

        self.action_dim = venv.action_space.n

        # The module consists of both parametric and non-parametric components.

        # There are two non-parametric components:
        #   episodic memory buffer 𝑴
        #   reward bonus estimation function 𝑩

        # memory buffer 𝑴
        self.ec_capacity = ec_capacity
        self.ec_memory_replacement_strategy = ec_memory_replacement_strategy
        self.ec_memory_reset_on_episode_end = ec_memory_reset_on_episode_end
        self.ec_memory = deque(maxlen=self.ec_capacity)

        # reward bonus estimation function 𝑩
        # α ∈ R+
        # β = 0.5 works well for fixed-duration episodes,
        # and β = 1 is preferred if an episode could have variable length.
        self.alpha = alpha
        self.beta = beta

        # bₙₒᵥₑₗₜᵧ
        # `After the bonus computation, the observation embedding is added to memory if the bonus b is larger
        #  than a novelty threshold bₙₒᵥₑₗₜᵧ`
        self.novelty_threshold = novelty_threshold

        # hyper-param for training
        self.gamma = gamma

        # There are two parametric components:
        #   embedding network 𝑬 : 𝒪 → ℝⁿ
        #   comparator network 𝑪 : ℝⁿ × ℝⁿ → [0, 1].

        # embedding network 𝑬 : 𝒪 → ℝⁿ
        self.embedding_feature_dim = embedding_feature_dim
        self.embedding_net = _create_fc_net(
            [self._observation_dim] + list(embedding_net_hiddens) + [self.embedding_feature_dim],
            embedding_net_activation,
            name="embedding_net",
        )

        # comparator network 𝑪 : ℝⁿ × ℝⁿ → [0, 1].
        self.comparator_net = _create_fc_net(
            [self.embedding_feature_dim * 2] + list(comparator_net_hiddens) + [1],
            comparator_net_activation,
            name="comparator_net",
        )
        # todo: better model
        layers = [
            SlimFC(
                in_size=self.embedding_feature_dim * 2,
                out_size=16,
                initializer=torch.nn.init.xavier_uniform_,
                activation_fn="relu",
            ),
            nn.Linear(
                in_features=16,
                out_features=1,
            ),
            nn.Sigmoid()
        ]
        self.comparator_net = nn.Sequential(*layers)

        # the aggregation function 𝑭 then maps the reachability buffer to [0.0, 1.0]
        self.similarity_aggregation = similarity_aggregation
        self.aggregation_function = self._construct_comparator_aggregation_function(
            similarity_aggregation=self.similarity_aggregation,
            percentile=similarity_aggregation_percentile,
            nth_largest_max=similarity_aggregation_nth_largest_max,
            relative_count_treshhold=similarity_aggregation_relative_count_threshold
        )

        embedding_params = list(self.embedding_net.parameters())
        comparator_params = list(self.comparator_net.parameters())

        self.lr = lr
        self._optimizer = torch.optim.Adam(embedding_params + comparator_params, lr=self.lr)

        # data training the networks
        initial_obs = self._process_obs_if_needed(self.venv.reset())

        # one trajectory per env
        # append new obs on every step
        self.trajectories = [
            [initial_obs[i]] for i in range(self.venv.num_envs)
        ]
        self.train_action_distance_threshold = train_action_distance_threshold

        # statistics
        self.n_postprocessings = 0
        self.global_stats = {
            "n_total_episodes": 0,
            "n_postprocessings": self.n_postprocessings,
            "memory_size": len(self.ec_memory)
        }
        self.sub_env_stats = [{
            "extrinsic_rewards": [],
            "intrinsic_rewards": [],
            "n_sub_env_episodes": 0,
        } for _ in range(self.venv.num_envs)]

    def calc_reward_bonus(self, similarity_score: float):
        # b = B(M, e) = α(β − C(M, e))
        return self.alpha * (self.beta - similarity_score)

    def _step_reward_bonus(self, observations: np.ndarray) -> np.ndarray:
        boni = np.zeros(self.venv.num_envs)
        with torch.no_grad():
            embedded_step_obs = self.embedding_net(
                torch.from_numpy(observations.astype(np.float32))
            ).cpu().detach()
            print(f"embedded_step_obs shape: {embedded_step_obs.shape}")

            if len(self.ec_memory):
                for i in range(self.venv.num_envs):
                    pairs = [torch.hstack((embedded_step_obs[i], memory_elem)) for memory_elem in self.ec_memory]
                    pairs = torch.stack(pairs)
                    # print(f"pairs shape: {pairs.shape}")
                    # `Then the similarity score between the memory buffer and the current embedding is computed
                    #  from the reachability buffer`
                    # not sure if the term 'reachability buffer' is 100% correct here,
                    # I call the output of the comparator net 'reachability buffer'
                    # (the intermediate result tht goes into the aggregation function)
                    # and the result of the aggregation function 'similarity_score'.
                    reachability_buffer = self.comparator_net(
                        pairs
                    ).ravel().cpu().detach().numpy()
                    # print(f"k_step_reachabilities: {reachability_buffer.shape}")

                    similarity_score = self.aggregation_function(reachability_buffer)
                    print(f"similarity_score: {similarity_score:.4f}")

                    bonus = self.calc_reward_bonus(similarity_score=similarity_score)

                    print(f"bonus: {bonus:.4f}")

                    # `After the bonus computation, the observation embedding is added to memory if the bonus b is
                    #  larger than a novelty threshold bₙₒᵥₑₗₜᵧ`
                    #
                    # if current observation is novel add bonus reward
                    # add obs to memory if novel
                    if bonus > self.novelty_threshold:
                        boni[i] = bonus
                        self.ec_memory.append(embedded_step_obs[i])
                    else:
                        pass

            else:
                for obs in embedded_step_obs:
                    self.ec_memory.append(obs)

        print(f"boni: {boni}")
        return boni

    def reset(self) -> VecEnvObs:
        observations = self.venv.reset()
        return observations

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()
        # return original obs at the end
        original_obs = observations
        observations = self._process_obs_if_needed(observations)

        # compare current observations with the ones in memory
        reward_bonus = np.zeros(self.venv.num_envs)

        # todo early stopping

        reward_bonus = self._step_reward_bonus(observations=observations)

        for i, done in enumerate(dones):
            # append observations to trajectories
            self.trajectories[i].append(observations[i])

            if done:
                post_proc_info = self._postprocess_trajectory(env_i=i)
                infos[i] = {**post_proc_info, **infos[i]}
                self.trajectories[i] = []

        augmented_rewards = rewards + reward_bonus

        extended_infos = self._extend_infos(
            augmented_rewards=augmented_rewards,
            original_rewards=rewards,
            intrinsic_rewards=reward_bonus,
            dones=dones,
            infos=infos,
        )

        return original_obs, augmented_rewards, dones, extended_infos

    def _extend_infos(self, augmented_rewards: np.ndarray, original_rewards: np.ndarray, intrinsic_rewards: np.ndarray,
                      dones: np.ndarray, infos) -> List[Dict]:

        self.global_stats["n_total_episodes"] += dones.sum()
        self.global_stats["n_postprocessings"] = self.n_postprocessings
        self.global_stats["_num_timesteps"] = self._num_timesteps
        self.global_stats["memory_size"] = len(self.ec_memory)

        print(f"memory_size: {len(self.ec_memory)}")

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

        return extended_infos

    def _postprocess_trajectory(self, env_i: int) -> Dict:
        self.n_postprocessings += 1
        trajectory = self.trajectories[env_i]
        training_data = []

        # `it predicts values close to 0 if probability of two observations being reach- able from one another within
        #  k steps is low, and values close to 1 when this probability is high`

        # print("*"*12)
        # positive examples (close distance)
        for k in range(1, self.train_action_distance_threshold + 1):
            obs_paris_of_distance_k = [
                # 1 = label
                [obs1, obs2, 1] for obs1, obs2 in zip(trajectory, trajectory[k:])
            ]
            training_data.extend(obs_paris_of_distance_k)

        # negative examples (large distance)
        k_start = self.train_action_distance_threshold * self.gamma
        k_end = k_start + self.train_action_distance_threshold
        for k in range(k_start, k_end):
            obs_paris_of_distance_k = [
                # 0 = label
                [obs1, obs2, 0] for obs1, obs2 in zip(trajectory, trajectory[k:])
            ]
            training_data.extend(obs_paris_of_distance_k)

        # shuffle training data
        # np.random.shuffle(training_data)  # shuffles first dim only

        # print(f"len training data: {len(training_data)}")
        x1 = self.embedding_net(
            torch.from_numpy(
                np.array(
                    [elem[0] for elem in training_data]
                ).astype(np.float32)
            )
        )
        # perform embedding of second observation
        x2 = self.embedding_net(
            torch.from_numpy(
                np.array(
                    [elem[1] for elem in training_data]
                ).astype(np.float32)
            )
        )
        y = torch.from_numpy(
            np.array(
                [elem[2] for elem in training_data]
            ).astype(np.float32)
        )
        # print(f"x1 shape: {x1.shape}")
        # print(f"x2 shape: {x2.shape}")
        # print(f"y shape: {y.shape}")

        # note: stack in the same order as used for calculating the reward bonus
        x_embedded_pairs = torch.hstack((x2, x1))
        # print(f"x_embedded_pairs shape: {x_embedded_pairs.shape}")

        k_step_reachability = self.comparator_net(
            x_embedded_pairs
        ).ravel()

        # print(f"k_step_reachability shape: {k_step_reachability.shape}")

        logistic_regression_loss = torch.nn.BCELoss()
        loss = logistic_regression_loss(k_step_reachability, y)

        # optimizer step
        #
        # zero grad before new step
        self._optimizer.zero_grad()
        # Backward pass and update
        loss.backward()
        self._optimizer.step()

        print(f"loss: {loss}")

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

    def _construct_comparator_aggregation_function(self,
                                                   similarity_aggregation: str,
                                                   percentile: int = 90,
                                                   nth_largest_max: int = 10,
                                                   relative_count_treshhold: float = 0.5) \
            -> Callable:
        # see episodic_curiosity.episodic_memory.similarity_to_memory
        # https://github.com/google-research/episodic-curiosity
        if similarity_aggregation not in ['max', 'nth_largest', 'percentile', 'relative_count']:
            raise ValueError(f"{similarity_aggregation} is not a valid argument. Valid options: "
                             f"'max', 'nth_largest', 'percentile', 'relative_count'.")

        def aggregation_function(reachability_buffer: np.ndarray, env_index: int = None) -> float:
            if similarity_aggregation == 'max':
                return np.max(reachability_buffer)
            elif similarity_aggregation == 'nth_largest':
                raise NotImplementedError("this aggregation is not implemented yet.")
            elif similarity_aggregation == 'percentile':
                if percentile is None:
                    raise ValueError(f"percentile (`similarity_aggregation_percentile`) "
                                     f"must be provided if `similarity_aggregation` is '{similarity_aggregation}'.")
                return np.percentile(reachability_buffer, percentile)
            elif similarity_aggregation == 'relative_count':
                # Number of samples in the memory similar to the input observation.
                if reachability_buffer is None:
                    raise ValueError(f"relative_count_treshhold (`similarity_aggregation_relative_count_treshhold`) "
                                     f"must be provided if `similarity_aggregation` is '{similarity_aggregation}'.")
                count = sum(reachability_buffer > relative_count_treshhold)
                return float(count) / len(reachability_buffer)

        return aggregation_function


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
    ec_model.learn(total_timesteps=budget)

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

    ec_venv = EpisodicCuriosityModuleWrapper(
        venv=venv,
    )
    ec_venv = VecMonitor(venv=ec_venv)
    ec_model = PPO('MlpPolicy', ec_venv, verbose=0)
    ec_model.learn(total_timesteps=budget)

    # mean_reward, std_reward = evaluate_policy(icm_model, icm_model.get_env(), n_eval_episodes=eval_episodes)
    # print(f"with icm: {mean_reward=}, {std_reward=}")
