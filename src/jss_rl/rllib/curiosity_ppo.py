# Import the RL algorithm (Trainer) we would like to use.
import pprint

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env

from jss_utils.PATHS import WANDB_API_KEY_FILE_PATH
from jss_utils.jss_logger import log
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv


def env_creator(env_config):
    return DisjunctiveGraphJssEnv(**env_config)  # return an env instance


register_env("GraphJsp-v0", env_creator)

jps_benchmark_instance_name = "ft06"
jps_instance = parser.get_instance_by_name(jps_benchmark_instance_name)
jps_instance_details = details.get_jps_instance_details(jps_benchmark_instance_name)

env_kwargs = {
    "jps_instance": jps_instance,
    "scaling_divisor": jps_instance_details["lower_bound"],
    "scale_reward": True,
    "normalize_observation_space": True,
    "flat_observation_space": True,
    "perform_left_shift_if_possible": True,
    "action_mode": 'task',
    "dtype": "float32",
    "verbose": 1,
    "default_visualisations": [
        "gantt_window",
        # "graph_window",  # very expensive
    ]
}
# Configure the algorithm.
config = {
    "log_level": 'WARN',
    "env": "GraphJsp-v0",
    "env_config": env_kwargs,
    "num_workers": 0,  # NOTE: must be 0 for Curiosity exploration
    "framework": "torch",  # torch
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    "evaluation_num_workers": 0,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
    "exploration_config": {
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
}

if __name__ == '__main__':
    log.info("RLlib basic ppo with curiosity exploration")

    trainer = PPOTrainer(config=config)

    log.info("training...")
    for i in range(2):
        train_data = trainer.train()
        log.info(pprint.pformat(train_data))

    log.info("evaluating...")
    eval_dict = trainer.evaluate()
    log.info(pprint.pformat(eval_dict))
    log.info("done")
