import itertools
from typing import List

import torch

import numpy as np
import torch.nn as nn
from ray.rllib.models.torch.misc import SlimFC


def build_dense_sequential_network(input_dim: int, output_dim: int, layers: List[int] = None, activation_function=None,
                                   scaled_output: bool = False):
    if layers is None:
        layers = []
    if activation_function is None:
        activation_function = nn.Tanh()
    # all dims
    dims = np.array([input_dim, *layers, output_dim])
    # in out pairs
    l_in_out = [torch.nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims, dims[1:])]
    # (linear, act function) pairs
    l_in_out = zip(
        l_in_out,
        [activation_function] * len(l_in_out) if not scaled_output
        else [ *list([activation_function] * (len(l_in_out)-1)), nn.Sigmoid()]
    )
    # flatten (list of tupels -> list)
    l_in_out = itertools.chain.from_iterable(l_in_out)
    return nn.Sequential(*l_in_out)






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