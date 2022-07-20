import pprint

import numpy as np
import wandb as wb

from typing import Dict

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_rl.rllib.episodic_curiosity_module import EpisodicCuriosity
from jss_rl.rllib.jss_action_mask_model import JssActionMaskModel
from jss_utils.PATHS import WANDB_API_KEY_FILE_PATH, WANDB_PATH
from jss_utils.jsp_env_utils import get_benchmark_instance_and_details

wb.tensorboard.patch(root_logdir=str(WANDB_PATH))


def env_creator(env_config):
    return DisjunctiveGraphJssEnv(**env_config)  # return an env instance


register_env("GraphJsp-v0", env_creator)
ModelCatalog.register_custom_model("jss_action_mask_model", JssActionMaskModel)


def run_experiment(
        jsp_instance: np.ndarray,
        jsp_instance_details: Dict,
        stop_conditions: Dict,
        num_samples: int,
        wandb_project: str) -> None:
    env_kwargs = {
        "jps_instance": jsp_instance,
        "scaling_divisor": jsp_instance_details["lower_bound"],
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "perform_left_shift_if_possible": True,
        "action_mode": 'task',
        "dtype": "float32",
        "verbose": 0,
        "env_transform": 'mask',  # for action masking
        "default_visualisations": [
            "gantt_window",
        ]
    }
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = 'WARN'
    config["env"] = "GraphJsp-v0"
    config["env_config"] = env_kwargs
    config["num_workers"] = 0
    config["framework"] = "torch"
    config["model"] = {
        "custom_model": "jss_action_mask_model",  # action masking
        "custom_model_config": {}
    }

    plain_ppo_config = config.copy()

    icm_config = config.copy()
    icm_config["exploration_config"] = {
        "type": "Curiosity",
        "eta": 0.25,
        "lr": 0.001,
        "feature_dim": 128,
        "feature_net_config": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [256, 256],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256, 256],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }

    ec_config = config.copy()
    ec_config["exploration_config"] = {
        "type": EpisodicCuriosity,
        "framework": "torch",
        "alpha": 1.0,
        "beta": 1.0,
        "embedding_dim": 128,
        "episodic_memory_capacity": 10,

        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }

    # for some reason the EC module combined with action masking crashes. (a NotImplementedError is raised somewhere...)
    for group, algo_config in zip(
            [
                "PPO",
                "PPO + ICM",
                # "PPO + EC"
            ],
            [
                plain_ppo_config,
                icm_config,
                # ec_config
            ]
    ):
        pprint.pprint(algo_config)
        tune.run(
            "PPO",
            checkpoint_freq=1,
            config=algo_config,
            stop=stop_conditions,
            num_samples=num_samples,
            callbacks=[
                WandbLoggerCallback(
                    project=wandb_project,
                    log_config=False,
                    group=group,
                    api_key_file=WANDB_API_KEY_FILE_PATH
                )
            ]
        )


if __name__ == '__main__':
    instance, details_dict = get_benchmark_instance_and_details(name="abz5")

    stop = {
        "training_iteration": 2,
        "episode_reward_mean": -1.02,
    }

    run_experiment(
        jsp_instance=instance,
        jsp_instance_details=details_dict,
        stop_conditions=stop,
        num_samples=1,
        wandb_project="jss_rllib_dev"
    )
