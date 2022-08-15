import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple, List, Dict, Any

from deepmd.embedding.embedding_net import Embedding_net
from deepmd.utils.network import embedding_net_rand_seed_shift
from deepmd.common import get_activation_func, get_precision
from deepmd.utils.type_embed import embed_atom_type
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION

lib_path = r"/root/code/deepmd-kit-pytorch-dev/source/build/op/libop_abi.so"
torch.ops.load_library(lib_path)


class ProdForceFunc(torch.autograd.Function):
    forward_op = torch.ops.prod_force.prod_force_se_a
    backward_op = torch.ops.prod_force.prod_force_se_a_grad

    @staticmethod
    def forward(ctx, net_deriv_reshape, descrpt_deriv, nlist, natoms, n_a_sel, n_r_sel):
        ctx.save_for_backward(net_deriv_reshape, descrpt_deriv, nlist, natoms)
        ctx.n_a_sel = n_a_sel
        ctx.n_r_sel = n_r_sel
        force = ProdForceFunc.forward_op(net_deriv_reshape, descrpt_deriv, nlist, natoms)
        return force

    @staticmethod
    def backward(ctx, grad_output):
        net_deriv_reshape, descrpt_deriv, nlist, natoms = ctx.saved_tensors
        grad_net = ProdForceFunc.backward_op(grad_output, net_deriv_reshape, descrpt_deriv,
                                         nlist, natoms, ctx.n_a_sel, ctx.n_r_sel)
        return grad_net, None, None, None, None, None


class ProdForceVirialFunc(torch.autograd.Function):
    forward_op = torch.ops.prod_virial.prod_virial_se_a
    backward_op = torch.ops.prod_virial.prod_virial_se_a_grad

    @staticmethod
    def forward(ctx, net_deriv_reshape, descrpt_deriv, rij, nlist, natoms, n_a_sel, n_r_sel):
        ctx.save_for_backward(net_deriv_reshape, descrpt_deriv, rij, nlist, natoms)
        ctx.n_a_sel = n_a_sel
        ctx.n_r_sel = n_r_sel
        force = ProdForceVirialFunc.forward_op(net_deriv_reshape, descrpt_deriv, rij, nlist, natoms)
        return force

    @staticmethod
    def backward(ctx, grad_output):
        net_deriv_reshape, descrpt_deriv, rij, nlist, natoms = ctx.saved_tensors
        
        virial, atom_virial = ProdForceVirialFunc.backward_op(grad_output, net_deriv_reshape, descrpt_deriv,
                                                              rij, nlist, natoms, ctx.n_a_sel, ctx.n_r_sel)
        return virial, None, atom_virial, None, None, None, None


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
        if rcut < rcut_smth:
            raise RuntimeError("rcut_smth (%f) should be no more than rcut (%f)!" % (rcut_smth, rcut))
        self.sel_a = sel
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
        self.filter_neuron = neuron
        self.n_axis_neuron = axis_neuron
        self.filter_resnet_dt = resnet_dt
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = embedding_net_rand_seed_shift(self.filter_neuron)
        self.trainable = trainable
        self.compress_activation_fn = get_activation_func(activation_function)
        self.filter_activation_fn = get_activation_func(activation_function)
        self.filter_precision = get_precision(precision)
        self.exclude_types = set()
        for tt in exclude_types:
            assert(len(tt) == 2)
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = set_davg_zero
        self.type_one_side = type_one_side

        # descrpt config
        self.sel_r = [ 0 for ii in range(len(self.sel_a)) ]
        self.ntypes = len(self.sel_a)
        assert(self.ntypes == len(self.sel_r))
        self.rcut_a = -1
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        self.useBN = False
        self.dstd = None
        self.davg = None
        self.compress = False
        self.embedding_net_variables = None
        self.mixed_prec = None
        self.place_holders = {}
        nei_type = np.array([])
        for ii in range(self.ntypes):
            nei_type = np.append(nei_type, ii * np.ones(self.sel_a[ii])) # like a mask 
        self.nei_type = torch.from_numpy(nei_type).to(torch.int32)
        self.prod_env_mat_a_forward = torch.ops.prod_env_mat.prod_env_mat_a

        avg_zero = np.zeros([self.ntypes, self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)
        std_ones = np.ones([self.ntypes, self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)

        self.avg_zero = torch.from_numpy(avg_zero)
        self.std_ones = torch.from_numpy(std_ones)

        self.original_sel = None
        self.stat_descrpt = None
        self.descrpt_reshape = None
        self.embedding_net = Embedding_net([2944, 1], self.filter_neuron, activation_fn=self.filter_activation_fn, ntypes=self.ntypes)

        self.rets = []

    def _compute_stat_descrpt(self, coord: torch.Tensor, type: torch.Tensor,
                              natoms_vec: torch.Tensor, box: torch.Tensor,
                              default_mesh: torch.Tensor) -> torch.Tensor:
        assert coord.dtype == GLOBAL_NP_FLOAT_PRECISION
        assert box.dtype == GLOBAL_NP_FLOAT_PRECISION

        assert type.dtype == torch.int32
        assert natoms_vec.dtype == torch.int32 and natoms_vec.shape[0] == self.ntypes+2
        assert default_mesh.dtype == torch.int32

        self.avg_zero.fill_(0)
        self.std_ones.fill_(1)

        self.stat_descrpt, descrpt_deriv, rij, nlist \
            = self.prod_env_mat_a_forward(coord, type, natoms_vec, box, default_mesh,
                                          self.avg_zero, self.std_ones,
                                          self.rcut_a, self.rcut_r, self.rcut_r_smth,
                                          self.sel_a, self.sel_r)
        
        return self.stat_descrpt

    def compute_input_stats(self,
                            data_coord : list, 
                            data_box : list, 
                            data_atype : list, 
                            natoms_vec : list,
                            mesh : list, 
                            input_dict : dict
    ) -> None:
        """
        Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.
        
        Parameters
        ----------
        data_coord
                The coordinates. Can be generated by deepmd.model.make_stat_input
        data_box
                The box. Can be generated by deepmd.model.make_stat_input
        data_atype
                The atom types. Can be generated by deepmd.model.make_stat_input
        natoms_vec
                The vector for the number of atoms of the system and different types of atoms. Can be generated by deepmd.model.make_stat_input
        mesh
                The mesh for neighbor searching. Can be generated by deepmd.model.make_stat_input
        input_dict
                Dictionary for additional input
        """
        all_davg = []
        all_dstd = []
        if True:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            for cc,bb,tt,nn,mm in zip(data_coord, data_box, data_atype, natoms_vec, mesh) :
                sysr,sysr2,sysa,sysa2,sysn \
                    = self._compute_dstats_sys_smth(cc,bb,tt,nn,mm)
                sumr.append(sysr)
                suma.append(sysa)
                sumn.append(sysn)
                sumr2.append(sysr2)
                suma2.append(sysa2)
            sumr = torch.sum(sumr, axis=0).cpu().numpy()
            suma = torch.sum(suma, axis=0).cpu().numpy()
            sumn = torch.sum(sumn, axis=0).cpu().numpy()
            sumr2 = torch.sum(sumr2, axis=0).cpu().numpy()
            suma2 = torch.sum(suma2, axis=0).cpu().numpy()
            for type_i in range(self.ntypes):
                davgunit = [sumr[type_i]/(sumn[type_i]+1e-15), 0, 0, 0]
                dstdunit = [self._compute_std(sumr2[type_i], sumr[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i])
                            ]
                davg = np.tile(davgunit, self.ndescrpt // 4)
                dstd = np.tile(dstdunit, self.ndescrpt // 4)
                all_davg.append(davg)
                all_dstd.append(dstd)

        if not self.set_davg_zero:
            self.davg = np.array(all_davg)
            self.davg = torch.from_numpy(self.davg)
        self.dstd = np.array(all_dstd)
        self.dstd = torch.from_numpy(self.dstd)
    
    def _compute_dstats_sys_smth(self,
                                 data_coord, 
                                 data_box, 
                                 data_atype,                             
                                 natoms_vec,
                                 mesh) :    
        dd_all = self._compute_stat_descrpt(data_coord, data_atype, natoms_vec, data_box, mesh)
        natoms = natoms_vec
        dd_all = dd_all.reshape([-1, self.ndescrpt * natoms[0]])
        start_index = 0
        sysr = []
        sysa = []
        sysn = []
        sysr2 = []
        sysa2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2+type_i]
            dd = dd_all[:, start_index:end_index]
            dd = dd.reshape([-1, self.ndescrpt])
            start_index = end_index        
            # compute
            dd = dd.reshape([-1, 4])
            ddr = dd[:,:1]
            dda = dd[:,1:]
            sumr = torch.sum(ddr)
            suma = torch.sum(dda) / 3.
            sumn = dd.shape[0]
            sumr2 = torch.sum(torch.multiply(ddr, ddr))
            suma2 = torch.sum(torch.multiply(dda, dda)) / 3.
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
        return sysr, sysr2, sysa, sysa2, sysn
    
    def _compute_std (self,sumv2, sumv, sumn) :
        if sumn == 0:
            return 1. / self.rcut_r
        val = np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val
    
    def _pass_filter(self, 
                     inputs,
                     atype,
                     natoms,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        type_embedding = None
        start_index = 0
        inputs = inputs.reshape([-1, natoms[0], self.ndescrpt])
        output = []
        output_qmat = []
        if not (self.type_one_side and len(self.exclude_types) == 0) and type_embedding is None:
            for type_i in range(self.ntypes):
                inputs_i = inputs[:, start_index: start_index + natoms[2 + type_i], :]
                inputs_i = inputs_i.reshape([-1, self.ndescrpt])

                layer, qmat = self._filter(inputs_i, type_i, name='filter_type_all'+suffix, natoms=natoms, reuse=reuse, trainable = trainable, activation_fn = self.filter_activation_fn, type_embedding=type_embedding)
                layer = layer.reshape([inputs.shape[0], natoms[2 + type_i], self.get_dim_out()])
                qmat  = qmat.reshape([inputs.shape[0], natoms[2 + type_i], self.get_dim_rot_mat_1() * 3])
                output.append(layer)
                output_qmat.append(qmat)
                start_index += natoms[2+type_i]
        output = torch.cat(output, axis = 1)
        output_qmat = torch.cat(output_qmat, axis = 1)
        return output, output_qmat

    def forward(self, 
                coord_ : torch.Tensor, 
                atype_ : torch.Tensor,
                natoms : torch.Tensor,
                box_ : torch.Tensor, 
                mesh : torch.Tensor,
                reuse : bool = None,
                suffix : str = ''
    ) -> torch.Tensor:
        davg = self.davg
        dstd = self.dstd
        if davg is None:
            davg = np.zeros([self.ntypes, self.ndescrpt]) 
        if dstd is None:
            dstd = np.ones ([self.ntypes, self.ndescrpt])

        t_rcut = torch.tensor(max([self.rcut_r, self.rcut_a]), dtype=GLOBAL_TF_FLOAT_PRECISION, requires_grad=False)
        t_ntypes = torch.tensor(self.ntypes, dtype=torch.int32, requires_grad=False)
        t_ndescrpt = torch.tensor(self.ndescrpt, dtype=torch.int32, requires_grad=False)
        t_sel = torch.tensor(self.sel_a, dtype=torch.int32, requires_grad=False)
        t_original_sel = torch.tensor(self.original_sel if self.original_sel is not None else self.sel_a, dtype=torch.int32, requires_grad=False)
        self.t_avg = torch.tensor(davg, dtype=GLOBAL_TF_FLOAT_PRECISION, requires_grad=False)
        self.t_std = torch.tensor(dstd, dtype=GLOBAL_TF_FLOAT_PRECISION, requires_grad=False)

        coord = coord_.reshape([-1, natoms[1] * 3])
        box = box_.reshape([-1, 9])
        atype = atype_.reshape([-1, natoms[1]])

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = self.forward_op(coord, atype, natoms, box, mesh,
                              self.t_avg,
                              self.t_std,
                              self.rcut_a,
                              self.rcut_r,
                              self.rcut_r_smth,
                              self.sel_a,
                              self.sel_r)
        self.descrpt_reshape = self.descrpt.reshape([-1, self.ndescrpt])

        self.dout, self.qmat = self._pass_filter(self.descrpt_reshape, 
                                                 atype,
                                                 natoms,
                                                 suffix = suffix, 
                                                 reuse = reuse, 
                                                 trainable = self.trainable)
        return self.dout

    def _filter(
        self, 
        inputs, 
        type_input,
        natoms,
        type_embedding = None,
        activation_fn=nn.Tanh, 
        stddev=1.0,
        bavg=0.0,
        name='linear', 
        reuse=None,
        trainable = True
    ):
        # natom x (nei x 4)
        shape = inputs.shape
        outputs_size = [1] + self.filter_neuron
        outputs_size_2 = self.n_axis_neuron
        all_excluded = all([(type_input, type_i) in self.exclude_types for type_i in range(self.ntypes)])
        if all_excluded:
            # all types are excluded so result and qmat should be zeros
            # we can safaly return a zero matrix...
            # See also https://stackoverflow.com/a/34725458/9567349
            # result: natom x outputs_size x outputs_size_2
            # qmat: natom x outputs_size x 3
            natom = inputs.shape()[0]
            result = torch.zeros((natom, outputs_size_2, outputs_size[-1]), dtype=GLOBAL_TF_FLOAT_PRECISION)
            qmat = torch.zeros((natom, outputs_size[-1], 3), dtype=GLOBAL_TF_FLOAT_PRECISION)
            return result, qmat
            

        start_index = 0
        type_i = 0
        # natom x 4 x outputs_size
        if type_embedding is None:
            rets = []
            for type_i in range(self.ntypes):
                ret = self._filter_lower(
                    type_i, type_input,
                    start_index, self.sel_a[type_i],
                    inputs,
                    natoms,
                    type_embedding = type_embedding,
                    is_exclude = (type_input, type_i) in self.exclude_types,
                    activation_fn = activation_fn,
                    stddev = stddev,
                    bavg = bavg,
                    trainable = trainable,
                    suffix = "_"+str(type_i))
                if (type_input, type_i) not in self.exclude_types:
                    # add zero is meaningless; skip
                    rets.append(ret)
                start_index += self.sel_a[type_i]
            # faster to use accumulate_n than multiple add
            xyz_scatter_1 = sum(rets)
        else :
            xyz_scatter_1 = self._filter_lower(
                type_i, type_input,
                start_index, np.cumsum(self.sel_a)[-1],
                inputs,
                natoms,
                type_embedding = type_embedding,
                is_exclude = False,
                activation_fn = activation_fn,
                stddev = stddev,
                bavg = bavg,
                trainable = trainable)

        # natom x nei x outputs_size
        # xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
        # natom x nei x 4
        # inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
        # natom x 4 x outputs_size
        # xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
        if self.original_sel is None:
            # shape[1] = nnei * 4
            nnei = shape[1] / 4
        else:
            nnei = torch.tensor(np.sum(self.original_sel), dtype=self.filter_precision, requires_grad=False)
        xyz_scatter_1 = xyz_scatter_1 / nnei
        # natom x 4 x outputs_size_2
        xyz_scatter_2 = xyz_scatter_1[:,:,0: outputs_size_2]
        # # natom x 3 x outputs_size_2
        # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = xyz_scatter_1[:, 1:4, :]
        # natom x outputs_size_1 x 3
        qmat = qmat.permute(0, 2, 1)
        # natom x outputs_size x outputs_size_2
        result = torch.bmm(xyz_scatter_1.permute(0, 2, 1), xyz_scatter_2)
        # natom x (outputs_size x outputs_size_2)
        result = result.reshape([-1, outputs_size_2 * outputs_size[-1]])

        return result, qmat


    def _filter_lower(
            self,
            type_i,
            type_input,
            start_index,
            incrs_index,
            inputs,
            natoms,
            type_embedding=None,
            is_exclude = False,
            activation_fn = None,
            bavg = 0.0,
            stddev = 1.0,
            trainable = True,
            suffix = '',
            center_atom_type_idx=0,
            nei_atom_type_idx=0
    ):
        """
        input env matrix, returns R.G
        """
        outputs_size = [1] + self.filter_neuron
        # cut-out inputs
        # with natom x (nei_type_i x 4)
        inputs_i = inputs[:, start_index * 4:start_index * 4 + incrs_index * 4]

        shape_i = inputs_i.shape
        natom = shape_i[0]
        # with (natom x nei_type_i) x 4
        inputs_reshape = inputs_i.reshape([-1, 4])
        # with (natom x nei_type_i) x 1
        xyz_scatter = inputs_reshape[:, 0:1].reshape([-1,1])

        # natom x 4 x outputs_size
        if (not is_exclude):
            # with (natom x nei_type_i) x out_size
            xyz_scatter = self.embedding_net(xyz_scatter, type_input, type_i)
            if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
        else:
            # we can safely return the final xyz_scatter filled with zero directly
            return torch.zeros((natom, 4, outputs_size[-1]), dtype=self.filter_precision)
        # natom x nei_type_i x out_size
        xyz_scatter = xyz_scatter.reshape((-1, shape_i[1]//4, outputs_size[-1]))
        # When using tf.reshape(inputs_i, [-1, shape_i[1]//4, 4]) below
        # [588 24] -> [588 6 4] correct
        # but if sel is zero
        # [588 0] -> [147 0 4] incorrect; the correct one is [588 0 4]
        # So we need to explicitly assign the shape to tf.shape(inputs_i)[0] instead of -1
        # natom x 4 x outputs_size
        return torch.bmm(inputs_i.reshape([natom, shape_i[1]//4, 4]).permute(0, 2, 1), xyz_scatter)

    def get_dim_out (self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.n_axis_neuron

    def get_dim_rot_mat_1 (self) -> int:
        """
        Returns the first dimension of the rotation matrix. The rotation is of shape dim_1 x 3
        """
        return self.filter_neuron[-1]


    def test_pass_filter(self, descrpt, atype, natoms):
        self.descrpt = descrpt
        self.descrpt_reshape = self.descrpt.reshape([-1, self.ndescrpt])

        self.dout, self.qmat = self._pass_filter(self.descrpt_reshape, 
                                                 atype,
                                                 natoms,
                                                 suffix = '', 
                                                 reuse = True, 
                                                 trainable = self.trainable)
        return self.dout, self.qmat
    

    def prod_force_virial(self, 
                          atom_ener : torch.Tensor, 
                          natoms : torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        net_deriv = torch.autograd.grad(atom_ener, self.descrpt_reshape).unsqueeze(0)

        net_deriv_reshape = net_deriv.reshape([-1, natoms[0] * self.ndescrpt])        
        force \
            = ProdForceFunc.apply(net_deriv_reshape,
                                          self.descrpt_deriv,
                                          self.nlist,
                                          natoms,
                                          n_a_sel = self.nnei_a,
                                          n_r_sel = self.nnei_r)
        virial, atom_virial \
            = ProdForceVirialFunc.apply(net_deriv_reshape,
                                           self.descrpt_deriv,
                                           self.rij,
                                           self.nlist,
                                           natoms,
                                           n_a_sel = self.nnei_a,
                                           n_r_sel = self.nnei_r)
        
        return force, virial, atom_virial

