import gym
import torch.nn as nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN


class JssActionMaskModel(TorchModelV2, nn.Module):
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
