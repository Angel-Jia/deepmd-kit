import torch
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
