import random
import sys
from collections import deque

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
    """
    todo:
    Implementation of:
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
            embedding_dim: int = 288,
            embedding_net_config: Optional[ModelConfigDict] = None,
            comparator_net_hiddens: Tuple[int] = (256,),
            comparator_net_activation: str = "relu",
            alpha: float = 1.0,
            beta: float = 1.0,
            lr: float = 1e-3,
            k: int = 2,
            gamma: int = 3,
            b_novelty: float = 0.0,
            episodic_memory_capacity: int = 100,
            clear_memory_every_episode: bool = True,
            sub_exploration: Optional[FromConfigSpec] = None,
            **kwargs
    ):
        """ todo
        """
        if not isinstance(action_space, (Discrete, MultiDiscrete)):
            raise ValueError(
                "Only (Multi)Discrete action spaces supported for Curiosity so far!"
            )

        super().__init__(action_space, model=model, framework=framework, **kwargs)

        if self.policy_config["num_workers"] != 0:
            raise ValueError(
                "Curiosity exploration currently does not support parallelism."
                " `num_workers` must be 0!"
            )

        self.embedding_dim = embedding_dim
        if embedding_net_config is None:
            embedding_net_config = self.policy_config["model"].copy()

        self.embedding_net_config = embedding_net_config
        self.comparator_net_hiddens = comparator_net_hiddens
        self.comparator_net_activation = comparator_net_activation

        self.action_dim = (
            self.action_space.n
            if isinstance(self.action_space, Discrete)
            else np.sum(self.action_space.nvec)
        )

        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.k = k
        self.gamma = gamma
        self.b_novelty = b_novelty
        self.clear_memory_every_episode = clear_memory_every_episode

        self.episodic_memory = deque(maxlen=episodic_memory_capacity)

        if sub_exploration is None:
            raise NotImplementedError
        self.sub_exploration = sub_exploration

        # Creates modules/layers inside the actual ModelV2.
        self._curiosity_embedding_net = ModelCatalog.get_model_v2(
            self.model.obs_space,
            self.action_space,
            self.embedding_dim,
            model_config=self.embedding_net_config,
            framework=self.framework,
            name="embedding_net",
        )

        self._comparator_net = self._create_fc_net(
            [2 * self.embedding_dim] + list(self.comparator_net_hiddens) + [1],
            self.comparator_net_activation,
            name="comparator_net",
        )

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
            embedding_params = list(self._curiosity_embedding_net.parameters())
            comparator_params = list(self._comparator_net.parameters())

            # Now that the Policy's own optimizer(s) have been created (from
            # the Model parameters (IMPORTANT: w/o(!) the curiosity params),
            # we can add our curiosity sub-modules to the Policy's Model.
            self.model._curiosity_embedding_net = self._curiosity_embedding_net.to(
                self.device
            )
            self.model._curiosity_comparator_net = self._comparator_net.to(
                self.device
            )
            self._optimizer = torch.optim.Adam(
                comparator_params + embedding_params, lr=self.lr
            )
        else:
            raise NotImplementedError(f"TF not implemented yet.")

        return optimizers

    @override(Exploration)
    def postprocess_trajectory(self, policy, sample_batch, tf_sess=None):
        if self.clear_memory_every_episode:
            self.episodic_memory.clear()

        if self.framework != "torch":
            self._postprocess_tf(policy, sample_batch, tf_sess)
        else:
            self._postprocess_torch(policy, sample_batch)

    def _postprocess_tf(self, policy, sample_batch, tf_sess):
        raise NotImplementedError(f"TF not implemented yet.")
        # return sample_batch

    def _postprocess_torch(self, policy, sample_batch):
        # perform embedding on observations
        embedded_obs, _ = self.model._curiosity_embedding_net(
            {
                SampleBatch.OBS:
                    torch.from_numpy(sample_batch[SampleBatch.OBS])
                        .to(policy.device),
            }
        )
        if not len(self.episodic_memory):
            self.episodic_memory.append(embedded_obs[0])

        pairs = torch.empty(size=(0, self.embedding_dim * 2), dtype=torch.float32)
        labels = torch.empty(size=(0, 1), dtype=torch.float32)

        # construct pairs

        # `it predicts values close to 0 if probability of two observations being reach- able from one another within
        #  k steps is low, and values close to 1 when this probability is high`

        # positive examples (close distance)
        for k in range(1, self.k + 1):
            for obs1, obs2 in zip(embedded_obs, embedded_obs[k:]):
                # x
                pair = torch.cat([obs1, obs2])
                pair = torch.reshape(pair, (1, self.embedding_dim * 2))
                pairs = torch.cat((pairs, pair), 0)
                # y
                label = torch.tensor(1)
                label = torch.reshape(label, (1, 1))
                labels = torch.cat((labels, label), 0)

        # negative examples (large distance)
        k_start = self.k * self.gamma
        k_end = k_start + self.k
        for k in range(k_start, k_end):
            for obs1, obs2 in zip(embedded_obs, embedded_obs[k:]):
                # x
                pair = torch.cat([obs1, obs2])
                pair = torch.reshape(pair, (1, self.embedding_dim * 2))
                pairs = torch.cat((pairs, pair), 0)
                # y
                label = torch.tensor(0)
                label = torch.reshape(label, (1, 1))
                labels = torch.cat((labels, label), 0)


        pred = self.model._curiosity_comparator_net(pairs)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(pred, labels)

        # Perform an optimizer step.
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


        # calculate boni

        boni = []

        with torch.no_grad():
            for single_embedded_observation in embedded_obs:
                # pair single_embedded_observation with all entries in episodic memory
                pairs = torch.empty(size=(0, self.embedding_dim * 2), dtype=torch.float32)

                for memory_entry in self.episodic_memory:
                    pair = torch.cat([memory_entry, single_embedded_observation])
                    pair = torch.reshape(pair, (1, self.embedding_dim * 2))
                    pairs = torch.cat((pairs, pair), 0)

                reachability_buffer = self.model._curiosity_comparator_net(pairs).cpu().detach().numpy()

                # aggregation

                percentile = 90
                similarity_score = np.percentile(reachability_buffer, percentile)

                # calc bonus b
                b = self._calc_reward_bonus(similarity_score=similarity_score)

                # `After the bonus computation, the observation embedding is added to memory if the bonus b is
                #  larger than a novelty threshold bₙₒᵥₑₗₜᵧ`
                #
                # if current observation is novel add b to boni (boni are added to the rewads after the loop)
                if b > self.b_novelty:
                    boni.append(b)
                    self.episodic_memory.append(single_embedded_observation)
                else:
                    boni.append(0.0)

        # add boni to rewards
        sample_batch[SampleBatch.REWARDS] = (
                sample_batch[SampleBatch.REWARDS]
                + np.array(boni)
        )

        return sample_batch

    def _calc_reward_bonus(self, similarity_score: float):
        # b = B(M, e) = α(β − C(M, e))
        return self.alpha * (self.beta - similarity_score)

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
