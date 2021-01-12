import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg, get_activation_func, get_precision, add_data_requirement
from deepmd.Network import one_layer
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.DescrptSeA import DescrptSeA
from deepmd.Network import embedding_net

class DescrptSeAEbd (DescrptSeA):
    def __init__ (self, jdata):
        DescrptSeA.__init__(self, jdata)
        args = ClassArg()\
               .add('type_nchanl',      int,    default = 4) \
               .add('type_nlayer',      int,    default = 2) \
               .add('type_one_side',    bool,   default = True) \
               .add('numb_aparam',      int,    default = 0)
        class_data = args.parse(jdata)
        self.type_nchanl = class_data['type_nchanl']
        self.type_nlayer = class_data['type_nlayer']
        self.type_one_side = class_data['type_one_side']
        self.numb_aparam = class_data['numb_aparam']
        if self.numb_aparam > 0:
            add_data_requirement('aparam', 3, atomic=True, must=True, high_prec=False)


    def build (self, 
               coord_, 
               atype_,
               natoms,
               box_, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):
        nei_type = np.array([])
        for ii in range(self.ntypes):
            nei_type = np.append(nei_type, ii * np.ones(self.sel_a[ii]))
        self.nei_type = tf.get_variable('t_nei_type', 
                                        [self.nnei],
                                        dtype = global_tf_float_precision,
                                        trainable = False,
                                        initializer = tf.constant_initializer(nei_type))
        self.dout = DescrptSeA.build(self, coord_, atype_, natoms, box_, mesh, input_dict, suffix = suffix, reuse = reuse)
        tf.summary.histogram('embedding_net_output', self.dout)

        return self.dout


    def _type_embed(self, 
                    atype,
                    ndim = 1,
                    reuse = None, 
                    suffix = '',
                    trainable = True):
        ebd_type = tf.cast(atype, self.filter_precision)
        ebd_type = ebd_type / float(self.ntypes)
        ebd_type = tf.reshape(ebd_type, [-1, ndim])
        for ii in range(self.type_nlayer):
            name = 'type_embed_layer_' + str(ii)
            ebd_type = one_layer(ebd_type,
                                 self.type_nchanl,
                                 activation_fn = self.filter_activation_fn,
                                 precision = self.filter_precision,
                                 name = name, 
                                 reuse = reuse,
                                 seed = self.seed + ii,
                                 trainable = trainable)
        name = 'type_embed_layer_' + str(self.type_nlayer)
        ebd_type = one_layer(ebd_type,
                             self.type_nchanl,
                             activation_fn = None,
                             precision = self.filter_precision,
                             name = name, 
                             reuse = reuse,
                             seed = self.seed + ii,
                             trainable = trainable)
        ebd_type = tf.reshape(ebd_type, [tf.shape(atype)[0], self.type_nchanl])
        return ebd_type            


    def _embedding_net(self, 
                       inputs,
                       natoms,
                       filter_neuron,
                       activation_fn=tf.nn.tanh, 
                       stddev=1.0,
                       bavg=0.0,
                       name='linear', 
                       reuse=None,
                       seed=None, 
                       trainable = True):
        '''
        inputs:  nf x na x (nei x 4)
        outputs: nf x na x nei x output_size
        '''
        # natom x (nei x 4)
        inputs = tf.reshape(inputs, [-1, self.ndescrpt])
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + filter_neuron
        with tf.variable_scope(name, reuse=reuse):
            xyz_scatter_total = []
            # with natom x (nei x 4)  
            inputs_i = inputs
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4])
            # with (natom x nei) x 1
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1])
            # with (natom x nei) x out_size
            xyz_scatter = embedding_net(xyz_scatter, 
                                        self.filter_neuron, 
                                        self.filter_precision, 
                                        activation_fn = activation_fn, 
                                        resnet_dt = self.filter_resnet_dt,
                                        stddev = stddev,
                                        bavg = bavg,
                                        seed = seed,
                                        trainable = trainable)
            # natom x nei x out_size
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))
            xyz_scatter_total.append(xyz_scatter)
        # natom x nei x outputs_size
        xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
        # nf x natom x nei x outputs_size
        xyz_scatter = tf.reshape(xyz_scatter, [tf.shape(inputs)[0], natoms[0], self.nnei, outputs_size[-1]])
        return xyz_scatter

    
    def _type_embedding_net_two_sides(self, 
                                      mat_g, 
                                      atype,
                                      natoms,
                                      name = '',
                                      reuse = None,
                                      seed = None,
                                      trainable = True):
        outputs_size = self.filter_neuron[-1]
        nframes = tf.shape(mat_g)[0]
        # (nf x natom x nei) x (outputs_size x chnl x chnl)
        mat_g = tf.reshape(mat_g, [nframes * natoms[0] * self.nnei, outputs_size])
        mat_g = one_layer(mat_g, 
                          outputs_size * self.type_nchanl * self.type_nchanl, 
                          activation_fn = None,
                          precision = self.filter_precision,
                          name = name+'_amplify',
                          reuse = reuse,
                          seed = self.seed,
                          trainable = trainable)        
        # nf x natom x nei x outputs_size x chnl x chnl
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], self.nnei, outputs_size, self.type_nchanl, self.type_nchanl])
        # nf x natom x outputs_size x chnl x nei x chnl
        mat_g = tf.transpose(mat_g, perm = [0, 1, 3, 4, 2, 5])
        # nf x natom x outputs_size x chnl x (nei x chnl)
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], outputs_size, self.type_nchanl, self.nnei * self.type_nchanl])
        
        # nei x nchnl
        ebd_nei_type = self._type_embed(self.nei_type, 
                                        reuse = reuse,
                                        trainable = True,
                                        suffix = '')
        # (nei x nchnl)
        ebd_nei_type = tf.reshape(ebd_nei_type, [self.nnei * self.type_nchanl])
        # (nframes x natom) x nchnl
        ebd_atm_type = self._type_embed(atype,
                                        reuse = True,
                                        trainable = True,
                                        suffix = '')    
        # (nframes x natom x nchnl)
        ebd_atm_type = tf.reshape(ebd_atm_type, [nframes * natoms[0] * self.type_nchanl])

        # nf x natom x outputs_size x chnl x (nei x chnl)
        mat_g = tf.multiply(mat_g, ebd_nei_type)
        # nf x natom x outputs_size x chnl x nei x chnl
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], outputs_size, self.type_nchanl, self.nnei, self.type_nchanl])
        # nf x natom x outputs_size x chnl x nei 
        mat_g = tf.reduce_mean(mat_g, axis = 5)
        # outputs_size x nei x nf x natom x chnl
        mat_g = tf.transpose(mat_g, perm = [2, 4, 0, 1, 3])
        # outputs_size x nei x (nf x natom x chnl)
        mat_g = tf.reshape(mat_g, [outputs_size, self.nnei, nframes * natoms[0] * self.type_nchanl])
        # outputs_size x nei x (nf x natom x chnl)
        mat_g = tf.multiply(mat_g, ebd_atm_type)
        # outputs_size x nei x nf x natom x chnl
        mat_g = tf.reshape(mat_g, [outputs_size, self.nnei, nframes, natoms[0], self.type_nchanl])
        # outputs_size x nei x nf x natom
        mat_g = tf.reduce_mean(mat_g, axis = 4)
        # nf x natom x nei x outputs_size
        mat_g = tf.transpose(mat_g, perm = [2, 3, 1, 0])        
        # (nf x natom) x nei x outputs_size
        mat_g = tf.reshape(mat_g, [nframes * natoms[0], self.nnei, outputs_size])
        return mat_g


    def _type_embedding_net_one_side(self, 
                                     mat_g, 
                                     atype,
                                     natoms,
                                     name = '',
                                     reuse = None,
                                     seed = None,
                                     trainable = True):
        outputs_size = self.filter_neuron[-1]
        nframes = tf.shape(mat_g)[0]
        # (nf x natom x nei) x (outputs_size x chnl x chnl)
        mat_g = tf.reshape(mat_g, [nframes * natoms[0] * self.nnei, outputs_size])
        mat_g = one_layer(mat_g, 
                          outputs_size * self.type_nchanl, 
                          activation_fn = None,
                          precision = self.filter_precision,
                          name = name+'_amplify',
                          reuse = reuse,
                          seed = self.seed,
                          trainable = trainable)        
        # nf x natom x nei x outputs_size x chnl
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], self.nnei, outputs_size, self.type_nchanl])
        # nf x natom x outputs_size x nei x chnl
        mat_g = tf.transpose(mat_g, perm = [0, 1, 3, 2, 4])
        # nf x natom x outputs_size x (nei x chnl)
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], outputs_size, self.nnei * self.type_nchanl])

        # nei x nchnl
        ebd_nei_type = self._type_embed(self.nei_type, 
                                        reuse = reuse,
                                        trainable = True,
                                        suffix = '')
        # (nei x nchnl)
        ebd_nei_type = tf.reshape(ebd_nei_type, [self.nnei * self.type_nchanl])

        # nf x natom x outputs_size x (nei x chnl)
        mat_g = tf.multiply(mat_g, ebd_nei_type)
        # nf x natom x outputs_size x nei x chnl
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], outputs_size, self.nnei, self.type_nchanl])
        # nf x natom x outputs_size x nei 
        mat_g = tf.reduce_mean(mat_g, axis = 4)
        # nf x natom x nei x outputs_size
        mat_g = tf.transpose(mat_g, perm = [0, 1, 3, 2])
        # (nf x natom) x nei x outputs_size
        mat_g = tf.reshape(mat_g, [nframes * natoms[0], self.nnei, outputs_size])
        return mat_g


    def _type_embedding_net_one_side_aparam(self, 
                                            mat_g, 
                                            atype,
                                            natoms,
                                            aparam,
                                            name = '',
                                            reuse = None,
                                            seed = None,
                                            trainable = True):
        outputs_size = self.filter_neuron[-1]
        nframes = tf.shape(mat_g)[0]
        # (nf x natom x nei) x (outputs_size x chnl x chnl)
        mat_g = tf.reshape(mat_g, [nframes * natoms[0] * self.nnei, outputs_size])
        mat_g = one_layer(mat_g, 
                          outputs_size * self.type_nchanl, 
                          activation_fn = None,
                          precision = self.filter_precision,
                          name = name+'_amplify',
                          reuse = reuse,
                          seed = self.seed,
                          trainable = trainable)        
        # nf x natom x nei x outputs_size x chnl
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], self.nnei, outputs_size, self.type_nchanl])
        # outputs_size x nf x natom x nei x chnl
        mat_g = tf.transpose(mat_g, perm = [3, 0, 1, 2, 4])
        # outputs_size x (nf x natom x nei x chnl)
        mat_g = tf.reshape(mat_g, [outputs_size, nframes * natoms[0] * self.nnei * self.type_nchanl])        
        # nf x natom x nnei        
        embed_type = tf.tile(tf.reshape(self.nei_type, [1, self.nnei]),
                             [nframes * natoms[0], 1])
        # (nf x natom x nnei) x 1
        embed_type = tf.reshape(embed_type, [nframes * natoms[0] * self.nnei, 1])        
        # nf x (natom x naparam)
        aparam = tf.reshape(aparam, [nframes, -1])
        # nf x natom x nnei x naparam        
        embed_aparam = op_module.map_aparam(aparam, self.nlist, natoms, n_a_sel = self.nnei_a, n_r_sel = self.nnei_r)
        # (nf x natom x nnei) x naparam
        embed_aparam = tf.reshape(embed_aparam, [nframes * natoms[0] * self.nnei, self.numb_aparam])
        # (nf x natom x nnei) x (naparam+1)
        embed_input = tf.concat((embed_type, embed_aparam), axis = 1)
        
        # (nf x natom x nnei) x nchnl
        ebd_nei_type = self._type_embed(embed_input, 
                                        ndim = self.numb_aparam + 1,
                                        reuse = reuse,
                                        trainable = True,
                                        suffix = '')
        # (nf x natom x nei x nchnl)
        ebd_nei_type = tf.reshape(ebd_nei_type, [nframes * natoms[0] * self.nnei * self.type_nchanl])

        # outputs_size x (nf x natom x nei x chnl)
        mat_g = tf.multiply(mat_g, ebd_nei_type)
        # outputs_size x nf x natom x nei x chnl
        mat_g = tf.reshape(mat_g, [outputs_size, nframes, natoms[0], self.nnei, self.type_nchanl])
        # outputs_size x nf x natom x nei 
        mat_g = tf.reduce_mean(mat_g, axis = 4)
        # nf x natom x nei x outputs_size
        mat_g = tf.transpose(mat_g, perm = [1, 2, 3, 0])
        # (nf x natom) x nei x outputs_size
        mat_g = tf.reshape(mat_g, [nframes * natoms[0], self.nnei, outputs_size])
        return mat_g


    def _pass_filter(self, 
                     inputs,
                     atype,
                     natoms,
                     input_dict,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        # nf x na x ndescrpt
        # nf x na x (nnei x 4)
        inputs = tf.reshape(inputs, [-1, natoms[0], self.ndescrpt])
        layer, qmat = self._ebd_filter(tf.cast(inputs, self.filter_precision), 
                                       atype,
                                       natoms,
                                       input_dict,
                                       name='filter_type_all'+suffix, 
                                       reuse=reuse, 
                                       seed = self.seed, 
                                       trainable = trainable, 
                                       activation_fn = self.filter_activation_fn)
        output      = tf.reshape(layer, [tf.shape(inputs)[0], natoms[0] * self.get_dim_out()])
        output_qmat = tf.reshape(qmat,  [tf.shape(inputs)[0], natoms[0] * self.get_dim_rot_mat_1() * 3])
        return output, output_qmat


    def _ebd_filter(self, 
                    inputs, 
                    atype,
                    natoms,
                    input_dict,
                    activation_fn=tf.nn.tanh, 
                    stddev=1.0,
                    bavg=0.0,
                    name='linear', 
                    reuse=None,
                    seed=None, 
                    trainable = True):
        outputs_size = self.filter_neuron[-1]
        outputs_size_2 = self.n_axis_neuron
        # nf x natom x (nei x 4)
        nframes = tf.shape(inputs)[0]
        shape = tf.reshape(inputs, [-1, self.ndescrpt]).get_shape().as_list()
        
        # nf x natom x nei x outputs_size        
        mat_g = self._embedding_net(inputs,
                                    natoms,
                                    self.filter_neuron,
                                    activation_fn = activation_fn, 
                                    stddev = stddev,
                                    bavg = bavg,
                                    name = name, 
                                    reuse = reuse,
                                    seed = seed,
                                    trainable = trainable)
        # nf x natom x nei x outputs_size        
        mat_g = tf.reshape(mat_g, [nframes, natoms[0], self.nnei, outputs_size])
        
        # (nf x natom) x nei x outputs_size
        if self.type_one_side:
            if self.numb_aparam > 0:
                aparam = input_dict['aparam']
                xyz_scatter \
                    = self._type_embedding_net_one_side_aparam(mat_g, 
                                                               atype,
                                                               natoms, 
                                                               aparam,
                                                               name = name,
                                                               reuse = reuse, 
                                                               seed = seed,
                                                               trainable = trainable)
            else:
                xyz_scatter \
                    = self._type_embedding_net_one_side(mat_g, 
                                                        atype,
                                                        natoms, 
                                                        name = name,
                                                        reuse = reuse, 
                                                        seed = seed,
                                                        trainable = trainable)
        else:
            xyz_scatter \
                = self._type_embedding_net_two_sides(mat_g, 
                                                     atype,
                                                     natoms, 
                                                     name = name,
                                                     reuse = reuse, 
                                                     seed = seed,
                                                     trainable = trainable)
        
        # natom x nei x 4
        inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
        # natom x 4 x outputs_size
        xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
        xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])
        # natom x 4 x outputs_size_2
        xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
        # # natom x 3 x outputs_size_2
        # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = tf.slice(xyz_scatter_1, [0,1,0], [-1, 3, -1])
        # natom x outputs_size_2 x 3
        qmat = tf.transpose(qmat, perm = [0, 2, 1])
        # natom x outputs_size x outputs_size_2
        result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a = True)
        # natom x (outputs_size x outputs_size_2)
        result = tf.reshape(result, [-1, outputs_size_2 * outputs_size])

        return result, qmat
