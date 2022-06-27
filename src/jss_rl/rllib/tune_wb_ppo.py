from ray import tune
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from ray.tune.integration.wandb import WandbLoggerCallback

from jss_utils.PATHS import WANDB_API_KEY_FILE_PATH
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from ray.tune import register_env


def env_creator(env_config):
    return DisjunctiveGraphJssEnv(**env_config)  # return an env instance


register_env("GraphJsp-v0", env_creator)

BENCHMARK_INSTANCE_NAME = "ta01"

jsp_instance = parser.get_instance_by_name(BENCHMARK_INSTANCE_NAME)
jsp_instance_details = details.get_jps_instance_details(BENCHMARK_INSTANCE_NAME)


if __name__ == '__main__':
    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "GraphJsp-v0"
    config["env_config"] = {
        "jps_instance": jsp_instance,
        "scaling_divisor": jsp_instance_details["lower_bound"],
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "perform_left_shift_if_possible": True,
        "action_mode": 'job',
        "dtype": "float32",
        "verbose": 0,
        "default_visualisations": [
            "gantt_window",
            # "graph_window",  # very expensive
        ]
    }

    config["num_workers"] = 0  # NOTE: must be 0 for Curiosity exploration
    config["exploration_config"] = {
        "type": "Curiosity",  # <- Use the Curiosity module for exploring.
        "eta": 0.1,  # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 288,  # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [256, 256],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256, 256],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }

    tune.run(
        "PPO",
        checkpoint_freq=1,
        config=config,
        stop={
            "timesteps_total": 500_000,
            "episode_reward_mean": -1.0,
        },
        callbacks=[WandbLoggerCallback(project="test", api_key_file=WANDB_API_KEY_FILE_PATH)]
    )
