import torch

import torch.nn as nn
from ray.rllib.models.torch.misc import SlimFC


def _create_fc_net(layer_dims, activation, name=None):
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
        layers = []

        for i in range(len(layer_dims) - 1):
            act = activation if i < len(layer_dims) - 2 else None
            layers.append(
                SlimFC(
                    in_size=layer_dims[i],
                    out_size=layer_dims[i + 1],
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=act,
                )
            )

        return nn.Sequential(*layers)