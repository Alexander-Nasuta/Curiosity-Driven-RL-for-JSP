from typing import Dict

import numpy as np
from ray.rllib import Policy, BaseEnv
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID

from jss_utils.jss_logger import log
from ray.rllib.agents.ppo import ppo


class MeanDistanceCallBack(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.deltas = []

    def on_postprocess_trajectory(
            self,
            *,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs
    ):
        pos = np.argmax(postprocessed_batch["obs"], -1)
        x, y = pos % 8, pos // 8
        self.deltas.extend((x ** 2 + y ** 2) ** 0.5)

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy], episode: Episode,
                       **kwargs) -> None:
        # https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        #
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        mean_distance_from_origin = np.mean(self.deltas)
        # print(f"mean. distance from origin={mean_distance_from_origin:.4}")
        # noinspection PyTypeChecker
        episode.custom_metrics["episode_mean_distance_from_origin"] = mean_distance_from_origin

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy], episode: Episode,
                         **kwargs) -> None:
        self.deltas = []

    def on_sample_end(self, *, worker, samples, **kwargs):
        # mean_distance_from_origin = np.mean(self.deltas)
        # print(f"mean. distance from origin={mean_distance_from_origin:.4}")
        # self.deltas = []
        pass


class FrozenlakeCallBack(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.deltas = []

    def on_postprocess_trajectory(
            self,
            *,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs
    ):
        pos = np.argmax(postprocessed_batch["obs"], -1)
        x, y = pos % 8, pos // 8
        self.deltas.extend((x ** 2 + y ** 2) ** 0.5)

    def on_sample_end(self, *, worker, samples, **kwargs):
        print(f"mean. distance from origin={np.mean(self.deltas):.4}")
        self.deltas = []


def run_icm_on_frozenlake():
    log.info("comparing icm performance with ppo algorithm on 'FrozenLake-v1' enviorment")
    config = ppo.DEFAULT_CONFIG.copy()
    # A very large frozen-lake that's hard for a random policy to solve
    # due to 0.0 feedback.
    config["env"] = "FrozenLake-v1"
    config["env_config"] = {
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
    # Print out observations to see how far we already get inside the Env.
    config["callbacks"] = FrozenlakeCallBack
    # Limit horizon to make it really hard for non-curious agent to reach
    # the goal state.
    config["horizon"] = 16
    # Local only.
    config["num_workers"] = 0
    config["lr"] = 0.001
    config["framework"] = "torch"

    config["exploration_config"] = {
        "type": "Curiosity",  # <- Use the Curiosity module for exploring.
        "framework": "torch",
        "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 288,  # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }
    config["callbacks"] = MeanDistanceCallBack

    trainer_with_icm = ppo.PPOTrainer(config=config)

    num_iterations = 10
    learnt = False
    for i in range(num_iterations):
        result = trainer_with_icm.train()
        if result["episode_reward_max"] > 0.0:
            log.info(f"reached goal after {i} iters!")
            learnt = True
            break
    trainer_with_icm.stop()

    assert learnt


if __name__ == '__main__':
    run_icm_on_frozenlake()
