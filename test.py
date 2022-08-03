import torch
import numpy as np

lib_path = r"/root/code/deepmd-kit-pytorch-dev/source/build/op/libop_abi.so"
torch.ops.load_library(lib_path)

names = ['coord', 'atype', 'natoms', 'box', 'mesh', 't_avg', 't_std',
         'descrpt', 'descrpt_deriv', 'rij', 'nlist']

rcut_a = -1.000000
rcut_r = 6.000000
rcut_r_smth = 0.500000
sel_a = torch.Tensor([46, 92]).to(torch.int32)
sel_r = torch.Tensor([0, 0]).to(torch.int32)

input_numpy = []
for i, name in enumerate(names):
    t = np.load(open('/root/code/deepmd-kit2.1.3/examples/water/se_e2_a/{}.npy'.format(name), 'rb'))
    tensor = torch.from_numpy(t).squeeze(0)
    # if tensor.dtype == torch.float64:
    #     tensor = tensor.to(torch.float32)
    if name != 'natoms' and name != 'box':
        tensor = tensor.cuda()
    print(name, tensor.dtype, tensor.shape)
    input_numpy.append(tensor)
    


ll = input_numpy[:7] + [rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r]
descrpt, descrpt_deriv, rij, nlist = torch.ops.ops_abi.prod_env_mat_a(*ll)
print(torch.sum(descrpt - input_numpy[-4]))
print(torch.sum(descrpt_deriv - input_numpy[-3]))
print(torch.sum(rij - input_numpy[-2]))
print(torch.sum(nlist - input_numpy[-1]))
print('----')
