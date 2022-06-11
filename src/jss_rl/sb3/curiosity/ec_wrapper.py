import sys

import torch
import numpy as np

from functools import reduce

import torch.nn as nn

from typing import List, Callable, Dict

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from jss_rl.sb3.curiosity.ec_memory import EpisodicMemory
from jss_rl.sb3.util.torch_dense_sequential_model_builder import build_dense_sequential_network


class EpisodicCuriosityEnvWrapper(VecEnvWrapper):

    def __init__(self,
                 venv: VecEnvWrapper,

                 alpha: float = 1.0,
                 beta: float = 0.5,
                 lr: float = 1e-3,
                 gamma: int = 2,
                 novelty_threshold: float = 0.0,

                 embedding_feature_dim: int = 288,
                 embedding_net_hiddens: List[int] = None,
                 embedding_net_activation=nn.Tanh(),

                 comparator_net_hiddens: List[int] = None,
                 comparator_net_activation=nn.Tanh(),
                 similarity_aggregation='percentile',  # 'max', 'nth_largest', 'percentile', 'relative_count'
                 similarity_aggregation_percentile: int = 90,
                 similarity_aggregation_nth_largest_max: int = 10,
                 similarity_aggregation_relative_count_threshold: float = 0.5,

                 ec_memory_replacement_strategy: str = 'random',
                 ec_capacity: int = 100,
                 ec_memory_reset_on_episode_end: bool = True,

                 train_action_distance_threshold: int = 2,
                 ):
        VecEnvWrapper.__init__(self, venv=venv)

        # default params
        if embedding_net_hiddens is None:
            embedding_net_hiddens = [64, 64]

        if comparator_net_hiddens is None:
            comparator_net_hiddens = [64, 64]

        # self._observation_dim = reduce((lambda x, y: x * y), venv.observation_space.shape)
        self._observation_dim = len(self.venv.reset()[0].ravel())

        # The module consists of both parametric and non-parametric components.

        # There are two non-parametric components:
        #   episodic memory buffer 𝑴
        #   reward bonus estimation function 𝑩

        # memory buffer 𝑴
        self.ec_memory_replacement_strategy = ec_memory_replacement_strategy
        self.ec_capacity = ec_capacity
        self.ec_memory_reset_on_episode_end = ec_memory_reset_on_episode_end
        self.ec_memory = EpisodicMemory(
            n_envs=venv.num_envs,
            capacity=self.ec_capacity,
            replacement_strategy=self.ec_memory_replacement_strategy,
            obs_shape=(embedding_feature_dim,)
        )
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
        self.embedding_net = build_dense_sequential_network(
            input_dim=self._observation_dim,
            output_dim=self.embedding_feature_dim,
            activation_function=embedding_net_activation,
            layers=embedding_net_hiddens
        )

        # comparator network 𝑪 : ℝⁿ × ℝⁿ → [0, 1].
        self.comparator_net = build_dense_sequential_network(
            input_dim=self.embedding_feature_dim * 2,
            output_dim=1,
            activation_function=comparator_net_activation,
            layers=comparator_net_hiddens,
            scaled_output=True  # each output elem between 0.0 and 1.0
        )

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
        initial_obs = self.venv.reset()

        # one trajectory per env
        # append new obs on every step
        self.trajectories = [np.array([initial_obs[i]]) for i in range(self.venv.num_envs)]
        self.train_action_distance_threshold = train_action_distance_threshold

        # for bonus return info
        self._bonus_rewards = [np.array([]) for _ in range(self.venv.num_envs)]
        # for extrinsic return info
        self._extrinsic_rewards = [np.array([]) for _ in range(self.venv.num_envs)]

    def calc_reward_bonus(self, similarity_score: float):
        # b = B(M, e) = α(β − C(M, e))
        return self.alpha * (self.beta - similarity_score)

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
                if nth_largest_max is None:
                    raise ValueError(f"nth_largest_max (`similarity_aggregation_nth_largest_max`) "
                                     f"must be provided if `similarity_aggregation` is '{similarity_aggregation}'.")
                if env_index is None:
                    raise ValueError(f"env_index must be provided if `similarity_aggregation` is 'nth_largest_max'.")
                n = min(nth_largest_max, self.ec_memory.len_of_sub_memory(env_index=env_index))
                flat = reachability_buffer.flatten()
                flat.sort()
                return flat[-n]
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

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()
        # `The episodic curiosity (EC) module takes the current observation o as input and produces a reward bonus b.`

        # compare current observations with the ones in memory
        reward_bonus = np.zeros(self.venv.num_envs)
        with torch.no_grad():
            print("asd")
            print(observations.shape)
            if self._observation_dim == 1:
                observations = np.array([np_array.ravel() for np_array in observations])
            print("asd2")
            print(observations.shape)

            embedded_obs = self.embedding_net(
                torch.from_numpy(observations.astype(np.float32))
            ).cpu().detach().numpy()

            print("asd3")
            if self.ec_memory.is_not_empty():
                memory_entries = self.ec_memory.get_all_entries()
                for env_i, (embedded_observation, memory_entries_of_sub_env) in enumerate(
                        zip(embedded_obs, memory_entries)):
                    obs_memory_pairs = np.array([
                        np.hstack((m_e, embedded_observation))
                        for m_e in memory_entries_of_sub_env
                    ])
                    # `Then the similarity score between the memory buffer and the current embedding is computed
                    #  from the reachability buffer`
                    # not sure if the term 'reachability buffer' is 100% correct here,
                    # I call the output of the comparator net 'reachability buffer'
                    # (the intermediate result tht goes into the aggregation function)
                    # and the result of the aggregation function 'similarity_score'.
                    reachability_buffer = self.comparator_net(
                        torch.from_numpy(obs_memory_pairs.astype(np.float32))
                    ).cpu().detach().numpy()
                    similarity_score = self.aggregation_function(reachability_buffer, env_index=env_i)
                    bonus = self.calc_reward_bonus(similarity_score)

                    # `After the bonus computation, the observation embedding is added to memory if the bonus b is
                    #  larger than a novelty threshold bₙₒᵥₑₗₜᵧ`
                    #
                    # if current observation is novel add bonus reward
                    # add obs to memory if novel
                    if bonus > self.novelty_threshold:
                        reward_bonus[env_i] = bonus
                        self.ec_memory.add_to_env_ec_memory(env_index=env_i, observation=embedded_observation)
                    else:
                        pass
            else:
                # if memory is empty
                self.ec_memory.add_all(embedded_obs)

        for i, done in enumerate(dones):
            # append observations to trajectories
            print("asd4")
            print(self.trajectories[i].shape)
            print(observations[i].shape)

            if self._observation_dim == 1:
                self.trajectories[i] = np.append(self.trajectories[i], observations[i], axis=0)
            else:
                self.trajectories[i] = np.append(self.trajectories[i], [observations[i]], axis=0)
            print("asd5")
            print(self.trajectories[i].shape)

            self._extrinsic_rewards[i] = np.append(self._extrinsic_rewards[i], [rewards[i]])
            self._bonus_rewards[i] = np.append(self._bonus_rewards[i], [reward_bonus[i]])

            infos[i]["extrinsic_reward"] = rewards[i]
            infos[i]['bonus_reward'] = reward_bonus[i]

            if done:
                infos[i]["extrinsic_return"] = self._extrinsic_rewards[i].sum()
                infos[i]['bonus_return'] = self._bonus_rewards[i].sum()
                infos[i]['total_return'] = infos[i]["extrinsic_return"] + infos[i]['bonus_return']

                self._extrinsic_rewards[i] = np.array([])
                self._bonus_rewards[i] = np.array([])

                post_proc_info = self._postprocess_trajectory(env_i=i)
                infos[i] = {**infos[i], **post_proc_info}
                # reset trajectory after postprocessing
                self.trajectories[i] = np.empty(shape=(0, *self.venv.observation_space.shape))

        augmented_reward = rewards + reward_bonus
        return observations, augmented_reward, dones, infos

    def reset(self) -> VecEnvObs:
        observations = self.venv.reset()
        self.trajectories = [np.array([observations[i]]) for i in range(self.venv.num_envs)]

        if self.ec_memory_reset_on_episode_end:
            self.ec_memory.reset_all()
        return observations

    def _postprocess_trajectory(self, env_i: int) -> Dict:
        print("post")
        trajectory = self.trajectories[env_i]
        training_data = np.empty(shape=(0, 3))

        # `it predicts values close to 0 if probability of two observations being reach- able from one another within
        #  k steps is low, and values close to 1 when this probability is high`

        # positive examples (close distance)
        for k in range(1, self.train_action_distance_threshold + 1):
            obs_paris_of_distance_k = np.array([
                # 0 = label
                [obs1, obs2, 1] for obs1, obs2 in zip(trajectory, trajectory[k:])
            ], dtype=object)
            training_data = np.append(training_data, obs_paris_of_distance_k, axis=0)

        # negative examples (large distance)
        k_start = self.train_action_distance_threshold * self.gamma
        k_end = k_start + self.train_action_distance_threshold
        for k in range(k_start, k_end):
            obs_paris_of_distance_k = np.array([
                # 0 = label
                [obs1, obs2, 0] for obs1, obs2 in zip(trajectory, trajectory[k:])
            ], dtype=object)
            training_data = np.append(training_data, obs_paris_of_distance_k, axis=0)

        # shuffle training data
        np.random.shuffle(training_data)  # shuffles first dim only

        x1 = np.vstack(training_data[:, 0]).astype(np.float32)
        x2 = np.vstack(training_data[:, 1]).astype(np.float32)
        y = torch.from_numpy(
            np.vstack(training_data[:, 2]).astype(np.float32)
        )

        if self._observation_dim == 1:
            x1 = np.array([np_array.ravel() for np_array in x1])
            x2 = np.array([np_array.ravel() for np_array in x2])

        # perform embedding of first observation
        x1 = self.embedding_net(
            torch.from_numpy(x1)
        )
        # perform embedding of second observation
        x2 = self.embedding_net(
            torch.from_numpy(x2)
        )
        # note: stack in the same order as used for calculating the reward bonus
        x_embedded_pairs = torch.hstack((x2, x1))

        k_step_reachability = self.comparator_net(
            x_embedded_pairs
        )

        logistic_regression_loss = torch.nn.BCELoss()
        loss = logistic_regression_loss(k_step_reachability, y)

        # optimizer step
        #
        # Backward pass and update
        loss.backward()
        self._optimizer.step()
        # zero grad before new step
        self._optimizer.zero_grad()

        print(f"loss : {loss.item()}")

        return {
            "ec_loss": loss.item()
        }


if __name__ == '__main__':
    from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3 import A2C

    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=4
    )

    budget = 5_000
    eval_episodes = 10

    cartpole_venv = VecMonitor(venv=venv)

    # model1 = A2C('MlpPolicy', cartpole_venv, verbose=0, seed=42)

    # model1.learn(total_timesteps=budget)
    # mean_reward, std_reward = evaluate_policy(model1, cartpole_venv, n_eval_episodes=eval_episodes)
    # print(f"without icm: {mean_reward=}, {std_reward=}")

    cartpole_venv.reset()
    cartpole_ec_venv = EpisodicCuriosityEnvWrapper(
        venv=venv,
    )
    cartpole_ec_venv = VecMonitor(venv=cartpole_ec_venv)

    model2 = A2C('MlpPolicy', cartpole_ec_venv, verbose=0, seed=42)
    # model2.set_env(cartpole_venv)
    model2.learn(total_timesteps=budget)

    mean_reward, std_reward = evaluate_policy(model2, model2.get_env(), n_eval_episodes=eval_episodes)

    print(f"with ec: {mean_reward=}, {std_reward=}")
