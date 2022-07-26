import pprint
import gym
import torch

import numpy as np
import torch.nn as nn

from typing import Dict

from ray import tune
from ray.tune import register_env

from ray.rllib.agents.ppo import ppo, PPOTrainer
from ray.rllib.models import ModelCatalog

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from jss_rl.rllib.jss_action_mask_model import JssActionMaskModel

from jss_utils.jss_logger import log
from jss_utils.jsp_env_utils import get_benchmark_instance_and_details


def env_creator(env_config):
    return DisjunctiveGraphJssEnv(**env_config)  # return an env instance


register_env("GraphJsp-v0", env_creator)
ModelCatalog.register_custom_model("jss_action_mask_model", JssActionMaskModel)


def run_mask_ppo_with_tune(jsp_instance: np.ndarray, jsp_instance_details: Dict) -> None:
    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "GraphJsp-v0"
    config["env_config"] = {
        "jps_instance": jsp_instance,
        "scaling_divisor": jsp_instance_details["lower_bound"],
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "perform_left_shift_if_possible": True,
        "action_mode": 'task',
        "env_transform": 'mask',
        "dtype": "float32",
        "verbose": 0,
        "default_visualisations": [
            "gantt_console",
            # "graph_window",  # very expensive
        ]
    }
    config["model"] = {
        "custom_model": "jss_action_mask_model",
        "custom_model_config": {}
    }
    config["framework"] = "torch"
    config["num_workers"] = 0

    tune.run(
        "PPO",
        checkpoint_freq=1,
        config=config,
        stop={
            "timesteps_total": 1_000,
            "episode_reward_mean": -1.0,
        },
    )


def run_mask_ppo_trainer(jsp_instance: np.ndarray, jsp_instance_details: Dict) -> None:
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["env"] = "GraphJsp-v0"
    config["env_config"] = {
        "jps_instance": jsp_instance,
        "scaling_divisor": jsp_instance_details["lower_bound"],
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "perform_left_shift_if_possible": True,
        "action_mode": 'task',
        "env_transform": 'mask',
        "dtype": "float32",
        "verbose": 0,
        "default_visualisations": [
            "gantt_console",
            # "graph_window",  # very expensive
        ]
    }
    config["framework"] = "torch"
    config["num_workers"] = 0
    config["model"] = {
        "custom_model": "jss_action_mask_model",
        "custom_model_config": {}
    }

    trainer = PPOTrainer(config=config)

    for i in range(1):
        train_data = trainer.train()
        log.info(pprint.pformat(train_data))


if __name__ == '__main__':
    instance, details_dict = get_benchmark_instance_and_details(name="abz5")
    # run_mask_ppo_with_tune(instance, details_dict)
    # run_mask_ppo_trainer(instance, details_dict)
