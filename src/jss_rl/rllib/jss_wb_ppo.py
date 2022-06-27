
# Import the RL algorithm (Trainer) we would like to use.
import pprint
from typing import Dict, OrderedDict
import gym
import torch.nn as nn
import torch

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.tune.integration.wandb import WandbLoggerCallback

from jss_utils.PATHS import WANDB_API_KEY_FILE_PATH
from jss_utils.jss_logger import log
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv



def env_creator(env_config):
    return DisjunctiveGraphJssEnv(**env_config)  # return an env instance


register_env("GraphJsp-v0", env_creator)


class CustomActionMaskModel(TorchModelV2, nn.Module):
    """
       Model that handles simple discrete action masking.
       This assumes the outputs are logits for a single Categorical action dist.
       PyTorch version of ActionMaskModel, derived from:
       https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py
       https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
       """

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, gym.spaces.dict.Dict)
        assert "action_mask" in orig_space.spaces
        assert "observations" in orig_space.spaces

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name,
            **kwargs,
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        # self.register_variables(self.internal_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model(
            {"obs": input_dict["obs"]["observations"]}
        )

        # Convert action_mask into a [0.0 || -inf] mask
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()



ModelCatalog.register_custom_model("action_mask_model", TorchModelV2)
ModelCatalog.register_custom_model("custom_action_mask_model", CustomActionMaskModel)


def main():
    BENCHMARK_INSTANCE_NAME = "abz5"

    jsp_instance = parser.get_instance_by_name(BENCHMARK_INSTANCE_NAME)
    jsp_instance_details = details.get_jps_instance_details(BENCHMARK_INSTANCE_NAME)

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

    config = {
        "log_level": 'WARN',
        "env": "GraphJsp-v0",
        "env_config": env_kwargs,
        "num_workers": 0,  # NOTE: must be 0 for Curiosity exploration
        "framework": "torch",  # torch
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [256, 256, 256],
            "fcnet_activation": "relu",
        },
        # Set up a separate evaluation worker set for the
        "evaluation_num_workers": 0,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
        "model": {
            "custom_model": "custom_action_mask_model", # action masking
            "custom_model_config": {}
        }

    }

    no_icm_config = config.copy()
    num_samples = 10

    stop = {
        "training_iteration": 10,
        "episode_reward_mean": -1.02,
    }


    tune.run(
        "PPO",
        checkpoint_freq=1,
        config=no_icm_config,
        stop=stop,
        num_samples=num_samples,
        callbacks=[
            WandbLoggerCallback(
                project="GraphJsp-Ray-test",
                log_config=False,
                group="PPO",
                api_key_file=WANDB_API_KEY_FILE_PATH
            )
        ]
    )

    icm_config = no_icm_config.copy()
    icm_config["exploration_config"] = {
        "type": "Curiosity",
        "eta": 0.25,
        "lr": 0.001,
        "feature_dim": 128,
        "feature_net_config": {
            "fcnet_hiddens": [256, 256, 256],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [256, 256, 256],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256, 256, 256],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
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
                project="GraphJsp-Ray-test",
                log_config=False,
                group="PPO + ICM",
                api_key_file=WANDB_API_KEY_FILE_PATH
            )
        ]
    )


if __name__ == '__main__':
    main()
