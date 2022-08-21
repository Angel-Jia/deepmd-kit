import torch
import torch.nn as nn
import numpy as np


class Embedding_net(nn.Module):
    def __init__(self, 
                  input_shape,
                  network_size,
                  activation_fn = nn.Tanh,
                  resnet_dt = False,
                  name_suffix = '',
                  stddev = 1.0,
                  bavg = 0.0,
                  seed = None,
                  uniform_seed = False,
                  initial_variables = None,
                  mixed_prec = None,
                  ntypes = 0
    ):
        super().__init__()
        self.outputs_size = [input_shape[1]] + network_size
        self.ntypes = ntypes
        self.layers = nn.ModuleDict()
        self.activation_fn = activation_fn()
        for center_atom_type_idx in range(self.ntypes):
            for nei_atom_type_idx in range(self.ntypes):
                layers = nn.ModuleList()
                for ii in range(1, len(self.outputs_size)):
                    layers.append(nn.Linear(self.outputs_size[ii - 1], self.outputs_size[ii]))
                self.layers['{}_{}'.format(center_atom_type_idx, nei_atom_type_idx)] = layers


    def forward(self, xx, center_atom_type_idx, nei_atom_type_idx):
        layers = self.layers['{}_{}'.format(center_atom_type_idx, nei_atom_type_idx)]
        for ii in range(1, len(self.outputs_size)):
            hidden = self.activation_fn(layers[ii - 1](xx))
            if self.outputs_size[ii] == self.outputs_size[ii-1]:
                xx += hidden
            elif self.outputs_size[ii] == self.outputs_size[ii-1] * 2:
                xx = torch.cat([xx, xx], 1) + hidden
            else:
                xx = hidden
        return xx


