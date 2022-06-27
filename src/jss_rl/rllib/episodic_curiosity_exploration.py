from gym.spaces import Discrete, MultiDiscrete, Space
import numpy as np
from typing import Optional, Tuple, Union

from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, MultiCategorical
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchMultiCategorical,
)
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder, one_hot as tf_one_hot
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
F = None
if nn is not None:
    F = nn.functional


class EpisodicCuriosity(Exploration):
    """Implementation of:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf

    Learns a simplified model of the environment based on three networks:
    1) Embedding observations into latent space ("feature" network).
    2) Predicting the action, given two consecutive embedded observations
    ("inverse" network).
    3) Predicting the next embedded obs, given an obs and action
    ("forward" network).

    The less the agent is able to predict the actually observed next feature
    vector, given obs and action (through the forwards network), the larger the
    "intrinsic reward", which will be added to the extrinsic reward.
    Therefore, if a state transition was unexpected, the agent becomes
    "curious" and will further explore this transition leading to better
    exploration in sparse rewards environments.
    """

    def __init__(
        self,
        action_space: Space,
        *,
        framework: str,
        model: ModelV2,
        feature_dim: int = 288,
        feature_net_config: Optional[ModelConfigDict] = None,
        inverse_net_hiddens: Tuple[int] = (256,),
        inverse_net_activation: str = "relu",
        forward_net_hiddens: Tuple[int] = (256,),
        forward_net_activation: str = "relu",
        beta: float = 0.2,
        eta: float = 1.0,
        lr: float = 1e-3,
        sub_exploration: Optional[FromConfigSpec] = None,
        **kwargs
    ):
        if not isinstance(action_space, Discrete):
            raise ValueError(
                "Only Discrete action spaces supported for Curiosity so far!"
            )

        super().__init__(action_space, model=model, framework=framework, **kwargs)

        if self.policy_config["num_workers"] != 0:
            raise ValueError(
                "Episodic Curiosity exploration currently does not support parallelism."
                " `num_workers` must be 0!"
            )

        # todo init NNs
        self.feature_dim = feature_dim
        if feature_net_config is None:
            feature_net_config = self.policy_config["model"].copy()
        self.feature_net_config = feature_net_config
        self.inverse_net_hiddens = inverse_net_hiddens
        self.inverse_net_activation = inverse_net_activation
        self.forward_net_hiddens = forward_net_hiddens
        self.forward_net_activation = forward_net_activation

        self.action_dim = (
            self.action_space.n
            if isinstance(self.action_space, Discrete)
            else np.sum(self.action_space.nvec)
        )

        # todo init NNs

        # This is only used to select the correct action
        self.exploration_submodule = from_config(
            cls=Exploration,
            config=self.sub_exploration,
            action_space=self.action_space,
            framework=self.framework,
            policy_config=self.policy_config,
            model=self.model,
            num_workers=self.num_workers,
            worker_index=self.worker_index,
        )

    @override(Exploration)
    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: Union[int, TensorType],
        explore: bool = True
    ):
        # Simply delegate to sub-Exploration module.
        return self.exploration_submodule.get_exploration_action(
            action_distribution=action_distribution, timestep=timestep, explore=explore
        )

    @override(Exploration)
    def get_exploration_optimizer(self, optimizers):
        # Create, but don't add Adam for curiosity NN updating to the policy.
        # If we added and returned it here, it would be used in the policy's
        # update loop, which we don't want (curiosity updating happens inside
        # `postprocess_trajectory`).
        if self.framework == "torch":
            feature_params = list(self._curiosity_feature_net.parameters())
            inverse_params = list(self._curiosity_inverse_fcnet.parameters())
            forward_params = list(self._curiosity_forward_fcnet.parameters())

            # Now that the Policy's own optimizer(s) have been created (from
            # the Model parameters (IMPORTANT: w/o(!) the curiosity params),
            # we can add our curiosity sub-modules to the Policy's Model.
            self.model._curiosity_feature_net = self._curiosity_feature_net.to(
                self.device
            )
            self.model._curiosity_inverse_fcnet = self._curiosity_inverse_fcnet.to(
                self.device
            )
            self.model._curiosity_forward_fcnet = self._curiosity_forward_fcnet.to(
                self.device
            )
            self._optimizer = torch.optim.Adam(
                forward_params + inverse_params + feature_params, lr=self.lr
            )
        else:
            raise NotImplementedError("Episodic Curiosity currently only supports torch.")

        return optimizers

    @override(Exploration)
    def postprocess_trajectory(self, policy, sample_batch, tf_sess=None):
        if self.framework != "torch":
            raise NotImplementedError("Episodic Curiosity currently only supports torch.")
        else:
            self._postprocess_torch(policy, sample_batch)


    def _postprocess_torch(self, policy, sample_batch):
        # todo
        return sample_batch

    def _create_fc_net(self, layer_dims, activation, name=None):
        """Given a list of layer dimensions (incl. input-dim), creates FC-net.

        Args:
            layer_dims (Tuple[int]): Tuple of layer dims, including the input
                dimension.
            activation: An activation specifier string (e.g. "relu").

        Examples:
            If layer_dims is [4,8,6] we'll have a two layer net: 4->8 (8 nodes)
            and 8->6 (6 nodes), where the second layer (6 nodes) does not have
            an activation anymore. 4 is the input dimension.
        """
        layers = (
            [tf.keras.layers.Input(shape=(layer_dims[0],), name="{}_in".format(name))]
            if self.framework != "torch"
            else []
        )

        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            if self.framework == "torch":
                layers.append(
                    SlimFC(
                        in_size=layer_dims[i],
                        out_size=layer_dims[i + 1],
                        initializer=torch.nn.init.xavier_uniform_,
                        activation_fn=act,
                    )
                )
            else:
                layers.append(
                    tf.keras.layers.Dense(
                        units=layer_dims[i + 1],
                        activation=get_activation_fn(act),
                        name="{}_{}".format(name, i),
                    )
                )

        if self.framework == "torch":
            return nn.Sequential(*layers)
        else:
            return tf.keras.Sequential(layers)



if __name__ == '__main__':
    import pprint
    from ray.rllib.agents.ppo import PPOTrainer

    config = {
        "log_level": 'WARN',
        "env": "CartPole-v1",
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
            "type": EpisodicCuriosity,
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

    trainer = PPOTrainer(config=config)
    print("training...")
    for i in range(2):
        train_data = trainer.train()
        print(pprint.pformat(train_data))