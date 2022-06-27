
# Import the RL algorithm (Trainer) we would like to use.
import pprint
from typing import Dict, OrderedDict

import gym
import torch.nn as nn
import torch

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN

from jss_utils.PATHS import WANDB_API_KEY_FILE_PATH
from jss_utils.jss_logger import log
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

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



def env_creator(env_config):
    return DisjunctiveGraphJssEnv(**env_config)  # return an env instance


register_env("GraphJsp-v0", env_creator)
ModelCatalog.register_custom_model("custom_action_mask_model", CustomActionMaskModel)
ModelCatalog.register_custom_model("action_mask_model", TorchModelV2)

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
    "verbose": 2,
    "env_transform": 'mask',
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
    },
    "model": {
        "custom_model": "action_mask_model",
        "custom_model_config": {}
    }

}

if __name__ == '__main__':
    log.info("RLlib mask ppo with curiosity exploration")

    trainer = PPOTrainer(config=config)

    log.info("training...")
    for i in range(2):
        train_data = trainer.train()
        log.info(pprint.pformat(train_data))

    log.info("evaluating...")
    eval_dict = trainer.evaluate()
    log.info(pprint.pformat(eval_dict))
    log.info("done")
