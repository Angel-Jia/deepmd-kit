#include "torch/torch.h"
#include "torch/extension.h"

#include "errors.h"
#include "prod_force.h"
#include "prod_force_grad.h"

template<typename FPTYPE>
static void
prod_force_se_a_generics(torch::Tensor &net_deriv_tensor, torch::Tensor &in_deriv_tensor,
                         torch::Tensor &nlist_tensor, torch::Tensor &natoms_tensor,
                         torch::Tensor &force_tensor, int &nloc, int &nall, int&nnei,
                         int &nframes, int &ndescrpt){
    // flat the tensors
    FPTYPE *p_force = force_tensor.data_ptr<FPTYPE>();
    const FPTYPE *p_net_deriv = net_deriv_tensor.data_ptr<FPTYPE>();
    const FPTYPE *p_in_deriv = in_deriv_tensor.data_ptr<FPTYPE>();
    const int *p_nlist = nlist_tensor.data_ptr<int>();

    int start_index = 0, end_index = nloc, nloc_loc = nloc;
    for(int64_t kk = 0; kk < nframes; ++kk){
        FPTYPE * force = p_force + kk * nall * 3;
        const FPTYPE * net_deriv = p_net_deriv + kk * nloc * ndescrpt;
        const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
        const int * nlist = p_nlist + kk * nloc * nnei;
        if(net_deriv_tensor.device().type() == torch::kCUDA) {
            #if GOOGLE_CUDA
            deepmd::prod_force_a_gpu_cuda(
                force, 
                net_deriv, in_deriv, nlist, nloc, nall, nnei);
            #endif // GOOGLE_CUDA
            
            #if TENSORFLOW_USE_ROCM
            deepmd::prod_force_a_gpu_rocm(
                force, 
                net_deriv, in_deriv, nlist, nloc, nall, nnei);
            #endif // TENSORFLOW_USE_ROCM
        }
        else if (net_deriv_tensor.device().type() == torch::kCPU) {
            deepmd::prod_force_a_cpu(
                force, 
                net_deriv, in_deriv, nlist, nloc_loc, nall, nnei, start_index=start_index);
            }
    }
}

template<typename FPTYPE>
static void
prod_force_se_a_grad_generics(torch::Tensor &grad_tensor, torch::Tensor &net_deriv_tensor,
                              torch::Tensor &in_deriv_tensor, torch::Tensor &nlist_tensor,
                              torch::Tensor &natoms_tensor, torch::Tensor &grad_net_tensor,
                              int &nframes, int &ndescrpt, int &nloc, int &nnei){
    // flat the tensors
    FPTYPE * p_grad_net = grad_net_tensor.data_ptr<FPTYPE>();
    const FPTYPE * p_grad = grad_tensor.data_ptr<FPTYPE>();
    const FPTYPE * p_net_deriv = net_deriv_tensor.data_ptr<FPTYPE>();
    const FPTYPE * p_in_deriv = in_deriv_tensor.data_ptr<FPTYPE>();
    const int * p_nlist	= nlist_tensor.data_ptr<int>();

    for (int64_t kk = 0; kk < nframes; ++kk){
        FPTYPE * grad_net = p_grad_net + kk * nloc * ndescrpt;
        const FPTYPE * grad = p_grad + kk * nloc * 3;
        const FPTYPE * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
        const int * nlist = p_nlist + kk * nloc * nnei; 
        if (grad_tensor.device().type() == torch::kCUDA) {
            #if GOOGLE_CUDA
            deepmd::prod_force_grad_a_gpu_cuda(    
                grad_net, 
                grad, in_deriv, nlist, nloc, nnei);
            #endif // GOOGLE_CUDA
        
            #if TENSORFLOW_USE_ROCM
            deepmd::prod_force_grad_a_gpu_rocm(    
                grad_net, 
                grad, in_deriv, nlist, nloc, nnei);
            #endif // TENSORFLOW_USE_ROCM
        }
        else if (grad_tensor.device().type() == torch::kCPU) {
            deepmd::prod_force_grad_a_cpu(    
                grad_net, 
                grad, in_deriv, nlist, nloc, nnei);
        }
    }
}

torch::Tensor prod_force_se_a(torch::Tensor net_deriv_tensor, torch::Tensor in_deriv_tensor,
                              torch::Tensor nlist_tensor, torch::Tensor natoms_tensor){
    // set size of the sample
    TORCH_CHECK((net_deriv_tensor.sizes().size() == 2), "Dim of net deriv should be 2");
    TORCH_CHECK((in_deriv_tensor.sizes().size() == 2), "Dim of input deriv should be 2");
    TORCH_CHECK((nlist_tensor.sizes().size() == 2), "Dim of nlist should be 2");
    TORCH_CHECK((natoms_tensor.sizes().size() == 1), "Dim of natoms should be 1");
    TORCH_CHECK((natoms_tensor.size(0) >= 3), "number of atoms should be larger than (or equal to) 3");

    TORCH_CHECK((natoms_tensor.dtype() == torch::kInt32), "Type of natoms should be int64");
    TORCH_CHECK(natoms_tensor.device().type() == torch::kCPU, "natoms_tensor should on cpu");
    auto natoms = natoms_tensor.flatten();
    int nloc = natoms[0].item<int>();
    int nall = natoms[1].item<int>();
    int nframes = net_deriv_tensor.size(0);
    int ndescrpt = net_deriv_tensor.size(1) / nloc;
    int nnei = nlist_tensor.size(1) / nloc;

    // check the sizes
    TORCH_CHECK((nframes == in_deriv_tensor.size(0)), "number of samples should match");
    TORCH_CHECK((nframes == nlist_tensor.size(0)), "number of samples should match");
    TORCH_CHECK((int64_t(nloc) * ndescrpt * 3 == in_deriv_tensor.size(1)), "number of descriptors should match");

    // Create an output tensor
    std::vector<int64_t> force_shape{nframes, 3 * nall};
    auto force_tensor_option = torch::TensorOptions().dtype(net_deriv_tensor.dtype()).layout(torch::kStrided)
                                                .device(net_deriv_tensor.device()).requires_grad(false);
    torch::Tensor force_tensor = torch::empty(force_shape, force_tensor_option);

    assert(nframes == force_shape[0]);
    assert(nframes == net_deriv_tensor.size(0));
    assert(nframes == in_deriv_tensor.size(0));
    assert(nframes == nlist_tensor.size(0));
    assert(nall * 3 == force_shape[1]);
    assert(nloc * ndescrpt == net_deriv_tensor.size(1));
    assert(nloc * ndescrpt * 3 == in_deriv_tensor.size(1));
    assert(nloc * nnei == nlist_tensor.size(1));
    assert(nnei * 4 == ndescrpt);

    if(net_deriv_tensor.dtype() == torch::kFloat32){
        prod_force_se_a_generics<float>(net_deriv_tensor, in_deriv_tensor,
                                        nlist_tensor, natoms_tensor,
                                        force_tensor, nloc, nall, nnei, nframes, ndescrpt);
    }else if(net_deriv_tensor.dtype() == torch::kFloat64){
        prod_force_se_a_generics<double>(net_deriv_tensor, in_deriv_tensor,
                                        nlist_tensor, natoms_tensor,
                                        force_tensor, nloc, nall, nnei, nframes, ndescrpt);
    }
    return force_tensor;
}

torch::Tensor prod_force_se_a_grad(torch::Tensor grad_tensor, torch::Tensor net_deriv_tensor,
                                    torch::Tensor in_deriv_tensor, torch::Tensor nlist_tensor,
                                    torch::Tensor natoms_tensor, int64_t n_a_sel, int64_t n_r_sel){
    int n_a_shift = n_a_sel * 4;
    // set size of the sample
    c10::IntArrayRef grad_shape = grad_tensor.sizes();
    c10::IntArrayRef net_deriv_shape = net_deriv_tensor.sizes();
    c10::IntArrayRef in_deriv_shape = in_deriv_tensor.sizes();
    c10::IntArrayRef nlist_shape = nlist_tensor.sizes();

    TORCH_CHECK((grad_shape.size() == 2), "Dim of grad should be 2");
    TORCH_CHECK((net_deriv_shape.size() == 2), "Dim of net deriv should be 2");
    TORCH_CHECK((in_deriv_shape.size() == 2), "Dim of input deriv should be 2");
    TORCH_CHECK((nlist_shape.size() == 2), "Dim of nlist should be 2");
    TORCH_CHECK((natoms_tensor.sizes().size() == 1), "Dim of natoms should be 1");
    TORCH_CHECK((natoms_tensor.dtype() == torch::kInt32), "Type of natoms should be int32");
    TORCH_CHECK((natoms_tensor.device().type() == torch::kCPU), "natoms_tensor should on cpu");

    TORCH_CHECK((natoms_tensor.size(0) >= 3), "number of atoms should be larger than (or equal to) 3");

    int nframes = net_deriv_tensor.size(0);
    int nloc = natoms_tensor.flatten()[0].item<int>();
    int ndescrpt = net_deriv_tensor.size(1) / nloc;
    int nnei = nlist_tensor.size(1) / nloc;

    // check the sizes
    TORCH_CHECK((nframes == grad_shape[0]), "number of frames should match");
    TORCH_CHECK((nframes == in_deriv_shape[0]), "number of frames should match");
    TORCH_CHECK((nframes == nlist_shape[0]), "number of frames should match");

    TORCH_CHECK((nloc * 3 == grad_shape[1]), "input grad shape should be 3 x natoms");
    TORCH_CHECK((nloc * ndescrpt * 3 == in_deriv_shape[1]), "number of descriptors should match");
    TORCH_CHECK((nnei == n_a_sel + n_r_sel), "number of neighbors should match");

    // Create an output tensor
    std::vector<int64_t> grad_net_shape{nframes, nloc * ndescrpt};
    auto grad_net_tensor_option = torch::TensorOptions().dtype(grad_tensor.dtype()).layout(torch::kStrided)
                                                        .device(grad_tensor.device()).requires_grad(false);
    torch::Tensor grad_net_tensor = torch::empty(grad_net_shape, grad_net_tensor_option);

    assert(nframes == grad_net_shape[0]);
    assert(nframes == grad_shape[0]);
    assert(nframes == net_deriv_tensor.size(0));
    assert(nframes == in_deriv_tensor.size(0));
    assert(nframes == nlist_tensor.size(0));
    assert(nloc * ndescrpt == grad_net_shape[1]);
    assert(nloc * 3 == grad_shape[1]);
    assert(nloc * ndescrpt == net_deriv_tensor.size(1));
    assert(nloc * ndescrpt * 3 == in_deriv_tensor.size(1));
    assert(nloc * nnei == nlist_tensor.size(1));
    assert(nnei * 4 == ndescrpt);

    if(grad_tensor.dtype() == torch::kFloat32){
        prod_force_se_a_grad_generics<float>(grad_tensor, net_deriv_tensor,
                                        in_deriv_tensor, nlist_tensor,
                                        natoms_tensor, grad_net_tensor,
                                        nframes, ndescrpt, nloc, nnei);
    }else if(grad_tensor.dtype() == torch::kFloat64){
        prod_force_se_a_grad_generics<double>(grad_tensor, net_deriv_tensor,
                                         in_deriv_tensor, nlist_tensor,
                                         natoms_tensor, grad_net_tensor,
                                         nframes, ndescrpt, nloc, nnei);
    }
    return grad_net_tensor;
}


TORCH_LIBRARY(prod_force, m){
    m.def("prod_force_se_a", prod_force_se_a);
    m.def("prod_force_se_a_grad", prod_force_se_a_grad);
}