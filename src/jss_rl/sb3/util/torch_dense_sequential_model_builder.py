import itertools
from typing import List

import torch

import numpy as np
import torch.nn as nn


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
