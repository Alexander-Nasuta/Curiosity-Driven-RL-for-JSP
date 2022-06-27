import sys
from typing import Dict

import numpy as np
import wandb

from ray import tune
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation import Episode
from ray.rllib.utils.exploration import Curiosity
from ray.rllib.utils.typing import PolicyID
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger, wandb_mixin
from ray.tune.integration.wandb import WandbLoggerCallback

from jss_utils.PATHS import WANDB_API_KEY_FILE_PATH
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from ray.tune import register_env


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


def main():
    project = "frozenlake-ray"
    config = ppo.DEFAULT_CONFIG.copy()
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
    #config["callbacks"] = MeanDistanceCallBack
    # Limit horizon to make it really hard for non-curious agent to reach
    # the goal state.
    config["horizon"] = 16
    # Local only.
    config["num_workers"] = 0
    config["lr"] = 0.001
    config["framework"] = "torch"
    config["callbacks"] = MeanDistanceCallBack

    no_icm_cofig = config.copy()

    num_samples = 10
    stop = {
            "training_iteration": 10,
            # "timesteps_total": 10_000,
            # "episode_reward_mean": sys.float_info.epsilon,
            "episode_reward_max": 1
    }

    tune.run(
        "PPO",
        checkpoint_freq=1,
        config=no_icm_cofig,
        stop=stop,
        num_samples=num_samples,
        callbacks=[
            WandbLoggerCallback(
                project=project,
                log_config=False,
                group="PPO",
                api_key_file=WANDB_API_KEY_FILE_PATH)
        ]
    )


    icm_config = config.copy()
    icm_config["exploration_config"] = {
        "type": "Curiosity",
        "eta": 0.2,
        "lr": 0.001,
        "feature_dim": 128,
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }

    tune.run(
        "PPO",
        checkpoint_freq=1,
        config=icm_config,
        stop=stop,
        num_samples=num_samples,
        callbacks=[
            WandbLoggerCallback(
                project=project,
                log_config=False,
                group="PPO + ICM (tuned)",
                api_key_file=WANDB_API_KEY_FILE_PATH)
        ]
    )


if __name__ == '__main__':
    main()