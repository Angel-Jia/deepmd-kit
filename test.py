import torch
import numpy as np

print('----')
lib_path = r"/root/code/deepmd-kit-pytorch-dev/source/build/op/libop_abi.so"
test_data_path = '/root/code/test_data/'
torch.ops.load_library(lib_path)

def mat_a_test():
    def test(gpu_test=False, float_test=False):
        op = torch.ops.prod_env_mat.prod_env_mat_a
        print('-->gpu test: {}, float test: {}'.format(gpu_test, float_test))

        input_numpy = []
        for i, name in enumerate(names):
            t = np.load(open('{}/mat_a/{}.npy'.format(test_data_path, name), 'rb'))
            tensor = torch.from_numpy(t).squeeze(0)
            if float_test and tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            if gpu_test and name != 'natoms' and name != 'box':
                tensor = tensor.cuda()
            print(name, tensor.dtype, tensor.shape)
            input_numpy.append(tensor)

        ll = input_numpy[:7] + [rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r]
        descrpt, descrpt_deriv, rij, nlist = op(*ll)
        print(torch.sum(descrpt - input_numpy[-4]))
        print(torch.sum(descrpt_deriv - input_numpy[-3]))
        print(torch.sum(rij - input_numpy[-2]))
        print(torch.sum(nlist - input_numpy[-1]))
        print('----')

    names = ['coord', 'atype', 'natoms', 'box', 'mesh', 't_avg', 't_std',
            'descrpt', 'descrpt_deriv', 'rij', 'nlist']
    
    rcut_a = -1.000000
    rcut_r = 6.000000
    rcut_r_smth = 0.500000
    sel_a = torch.Tensor([46, 92]).to(torch.int32)
    sel_r = torch.Tensor([0, 0]).to(torch.int32)

    test()
    test(gpu_test=True)
    test(gpu_test=True, float_test=True)


def force_test():
    def test_foce(gpu_test=False, float_test=False, op=None):
        print('-->gpu test: {}, float test: {}'.format(gpu_test, float_test))
        op = torch.ops.prod_force.prod_force_se_a
        names = ['net_deriv_reshape', 'descrpt_deriv', 'nlist',
                 'natoms', 'force']

        input_numpy = []
        for i, name in enumerate(names):
            t = np.load(open('{}/force/{}.npy'.format(test_data_path, name), 'rb'))
            tensor = torch.from_numpy(t).squeeze(0)
            if float_test and tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            if gpu_test and name != 'natoms':
                tensor = tensor.cuda()
            print(name, tensor.dtype, tensor.shape)
            input_numpy.append(tensor)

        ll = input_numpy[:4]
        force = op(*ll)
        print(torch.sum(force - input_numpy[-1]))
        print('----')
    
    test_foce()
    test_foce(gpu_test=True)
    test_foce(gpu_test=True, float_test=True)


def virial_test():
    def test_foce(gpu_test=False, float_test=False, op=None):
        print('-->gpu test: {}, float test: {}'.format(gpu_test, float_test))
        op = torch.ops.prod_virial.prod_virial_se_a
        names = ['net_deriv_reshape', 'descrpt_deriv', 'rij', 'nlist',
                 'natoms', 'virial', 'atom_virial']

        input_numpy = []
        for i, name in enumerate(names):
            t = np.load(open('{}/force/{}.npy'.format(test_data_path, name), 'rb'))
            tensor = torch.from_numpy(t).squeeze(0)
            if float_test and tensor.dtype == torch.float64:
                tensor = tensor.to(torch.float32)
            if gpu_test and name != 'natoms':
                tensor = tensor.cuda()
            print(name, tensor.dtype, tensor.shape)
            input_numpy.append(tensor)

        ll = input_numpy[:5]
        virial, atom_virial = op(*ll)
        print(torch.sum(virial - input_numpy[-2]))
        print(torch.sum(atom_virial - input_numpy[-1]))
        print('----')
    
    test_foce()
    test_foce(gpu_test=True)
    test_foce(gpu_test=True, float_test=True)
# mat_a_test()
# force_test()
virial_test()

