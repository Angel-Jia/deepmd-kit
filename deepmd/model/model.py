from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Dict, Any
import numpy as np

from ..descriptor.se_a import DescrptSeA
from ..fit.ener import EnerFitting

class EnergyModel(nn.Module):
    def __init__(self) -> None:
        super(EnergyModel, self).__init__()
        rcut = 6.0
        rcut_smth = 0.5
        sel = [46, 92]
        neuron = [25, 50, 100]
        axis_neuron = 16
        resnet_dt = False
        trainable = True
        seed = 1
        type_one_side = False
        exclude_types = []
        set_davg_zero = False
        activation_function = 'tanh'
        precision = 'default'
        uniform_seed = False
        self.descript_model = DescrptSeA(
                                        rcut,
                                        rcut_smth,
                                        sel,
                                        neuron,
                                        axis_neuron,
                                        resnet_dt,
                                        trainable,
                                        seed,
                                        type_one_side,
                                        exclude_types,
                                        set_davg_zero,
                                        activation_function,
                                        precision,
                                        uniform_seed
                                    )
        neuron = [240, 240, 240]
        resnet_dt = True
        numb_fparam = 0
        numb_aparam = 0
        rcond = 0.001
        tot_ener_zero = False
        trainable = True
        seed = 1
        atom_ener = []
        activation_function = 'tanh'
        precision = 'default'
        uniform_seed = False

        dim = 1600
        ntypes = 2
        self.fitting_net = EnerFitting(ntypes, dim, neuron, resnet_dt, numb_fparam, numb_aparam, rcond, tot_ener_zero, seed, atom_ener, activation_function, precision, uniform_seed)

        self.dout = None
    
    def forward(self, 
                coord_ : torch.Tensor, 
                atype_ : torch.Tensor,
                natoms : torch.Tensor,
                box_ : torch.Tensor, 
                mesh : torch.Tensor,
                input_dict: dict
    ) -> torch.Tensor:
    
        coord = coord_.reshape([-1, natoms[1] * 3])
        atype = atype_.reshape([-1, natoms[1]])
        input_dict['nframes'] = coord.shape[0]

        dout = self.descript_model(coord_, atype_, natoms, box_, mesh, True, '')
        atom_ener = self.fitting_net(dout, natoms)

        energy_raw = atom_ener

        energy_raw = energy_raw.reshape([-1, natoms[0]])
        energy = torch.sum(energy_raw, dim=1)

        force, virial, atom_virial \
            = self.descript_model.prod_force_virial(atom_ener, natoms)
        
        force = force.reshape([-1, 3 * natoms[1]])

        virial = virial.reshape([-1, 9])
        atom_virial = atom_virial.reshape([-1, 9 * natoms[1]])
        

        model_dict = {}
        model_dict['energy'] = energy
        model_dict['force'] = force
        model_dict['virial'] = virial
        model_dict['atom_ener'] = energy_raw
        model_dict['atom_virial'] = atom_virial
        model_dict['coord'] = coord
        model_dict['atype'] = atype
        
        return model_dict
        

    


    




