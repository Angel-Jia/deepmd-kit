import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Dict, Any
import numpy as np

from ..descriptor.se_a import MatAFunc, ProdForceFunc, ProdForceVirialFunc

class DescrptSeA(nn.Module):
    def __init__(self, 
                 rcut: float,
                 rcut_smth: float,
                 sel: List[str],
                 neuron: List[int] = [24,48,96],
                 axis_neuron: int = 8,
                 resnet_dt: bool = False,
                 trainable: bool = True,
                 seed: int = None,
                 type_one_side: bool = True,
                 exclude_types: List[List[int]] = [],
                 set_davg_zero: bool = False,
                 activation_function: str = 'tanh',
                 precision: str = 'default',
                 uniform_seed: bool = False
    ) -> None:
        """
        Constructor
        """
        super(DescrptSeA, self).__init__()
        self.t_avg = None
        self.t_std = None
    




