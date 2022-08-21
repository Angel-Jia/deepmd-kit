from turtle import forward
import torch
import torch.nn as nn

import numpy as np
from typing import Tuple, List

from deepmd.common import get_activation_func, get_precision

class IdtLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layer = nn.parameter.Parameter(torch.rand(input_size, requires_grad=True))

    def forward(self, inputs):
        return inputs * self.layer


class EnerFitting(nn.Module):
    def __init__ (self, 
                  ntypes : int,
                  dim_descrpt: int,
                  neuron : List[int] = [120,120,120],
                  resnet_dt : bool = True,
                  numb_fparam : int = 0,
                  numb_aparam : int = 0,
                  rcond : float = 1e-3,
                  tot_ener_zero : bool = False,
                  seed : int = None,
                  atom_ener : List[float] = [],
                  activation_function : str = 'tanh',
                  precision : str = 'default',
                  uniform_seed: bool = False
    ) -> None:
        super().__init__()
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.rcond = rcond
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.fitting_activation_fn = get_activation_func(activation_function)()
        self.fitting_precision = get_precision(precision)
        self.useBN = False
        self.bias_atom_e = np.zeros(self.ntypes, dtype=np.float64)
        self.fitting_net_variables = None
        self.mixed_prec = None

        self.fparam = None
        self.aparam = None

        self.input_size = [dim_descrpt] + self.n_neuron
        self.output_size = self.n_neuron + [1]

        self.layers = nn.ModuleDict()
        self._build_lowers_layers()

    
    def _build_lowers_layers(self):
        for type_i in range(self.ntypes):
            layers = nn.ModuleList()
            idt_layers = nn.ModuleList()
            layers.append(nn.Linear(self.input_size[0], self.output_size[0]))
            for ii in range(1, len(self.n_neuron)):
                assert self.n_neuron[ii] == self.n_neuron[ii-1]
                layers.append(nn.Linear(self.input_size[ii], self.output_size[ii]))
                idt_layers.append(IdtLayer(self.output_size[ii]))
            
            layers.append(nn.Linear(self.input_size[-1], self.output_size[-1]))
            self.layers['type_{}'.format(type_i)] = layers
            self.layers['type_{}_idt'.format(type_i)] = idt_layers
    

    def forward(self,
                inputs: torch.Tensor,
                natoms : torch.Tensor):
        inputs = inputs.reshape([-1, natoms[0], self.dim_descrpt])
        bias_atom_e = self.bias_atom_e
        if self.bias_atom_e is not None :
            assert(len(self.bias_atom_e) == self.ntypes)
        
        start_index = 0
        outs_list = []
        for type_i in range(self.ntypes):
            
            type_bias_ae = bias_atom_e[type_i]
            final_layer = self._lowers_layers_forward(
                start_index, natoms[2+type_i], 
                inputs, self.fparam, self.aparam, 
                bias_atom_e=type_bias_ae)
            final_layer = final_layer.reshape([inputs.shape[0], natoms[2+type_i]])
            outs_list.append(final_layer)
            start_index += natoms[2+type_i]
        # concat the results
        # concat once may be faster than multiple concat
        outs = torch.cat(outs_list, axis = 1)
        return outs.reshape([-1])

    
    def _lowers_layers_forward(
            self,
            start_index,
            natoms,
            inputs,
            fparam = None,
            aparam = None, 
            bias_atom_e = 0.0,
            type_i = 0
    ):
        inputs_i = inputs[:, start_index: start_index + natoms, :]
        inputs_i = inputs_i.reshape([-1, self.dim_descrpt])

        layers = self.layers['type_{}'.format(type_i)]
        idt_layers = self.layers['type_{}_idt'.format(type_i)]

        inputs_i = self.fitting_activation_fn(layers[0](inputs_i))
        for ii in range(1, len(self.n_neuron)):
            assert self.n_neuron[ii] == self.n_neuron[ii-1]
            inputs_i = inputs_i + self.fitting_activation_fn(idt_layers[ii - 1](layers[ii](inputs_i)))
        
        inputs_i = layers[-1](inputs_i) + bias_atom_e
        if (not self.uniform_seed) and (self.seed is not None): self.seed += 3

        return inputs_i



