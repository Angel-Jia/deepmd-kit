#include "torch/torch.h"
#include "torch/extension.h"

#include "utilities.h"
#include "coord.h"
#include "region.h"
#include "neighbor_list.h"
#include "prod_env_mat.h"
#include "errors.h"


#if GOOGLE_CUDA
template<typename FPTYPE>
static int
_norm_copy_coord_gpu(
    torch::Tensor & descrpt_tensor,
    torch::Tensor * tensor_list,
    FPTYPE * & coord_cpy,
    int * & type_cpy,
    int * & idx_mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r);

template<typename FPTYPE>
static int
_build_nlist_gpu(
    torch::Tensor & descrpt_tensor,
    torch::Tensor * tensor_list,
    int * &ilist, 
    int * &numneigh,
    int ** &firstneigh,
    int * &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r);

static void
_map_nlist_gpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei);

template <typename FPTYPE>
static void
_prepare_coord_nlist_gpu(
    torch::Tensor & descrpt_tensor,
    torch::Tensor * tensor_list,
    FPTYPE const ** coord,
    FPTYPE * & coord_cpy,
    int const** type,
    int * & type_cpy,
    int * & idx_mapping,
    deepmd::InputNlist & inlist,
    int * & ilist,
    int * & numneigh,
    int ** & firstneigh,
    int * & jlist,
    int * & nbor_list_dev,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int mesh_tensor_size,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial);
#endif //GOOGLE_CUDA

template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    torch::Tensor & descrpt_tensor,
    FPTYPE const ** coord,
    std::vector<FPTYPE> & coord_cpy,
    int const** type,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    deepmd::InputNlist & inlist,
    std::vector<int> & ilist,
    std::vector<int> & numneigh,
    std::vector<int*> & firstneigh,
    std::vector<std::vector<int>> & jlist,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial);

static void
_map_nlist_cpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei);

template<typename FPTYPE>
static void 
_prod_env_mat_a_generics(torch::Tensor& descrpt_tensor, torch::Tensor& descrpt_deriv_tensor,
                        torch::Tensor& rij_tensor, torch::Tensor& nlist_tensor,
                        torch::Tensor& coord_tensor, torch::Tensor& type_tensor,
                        torch::Tensor& natoms_tensor, torch::Tensor& box_tensor,
                        torch::Tensor& mesh_tensor,
                        torch::Tensor& avg_tensor, torch::Tensor& std_tensor,
                        int & nsamples, int & nloc, int & ndescrpt, int & nnei, int & nall,
                        int *& nbor_list_dev, int & mem_cpy, int & mem_nnei, int & max_nbor_size,
                        float & rcut_r_smth,
                        int & nei_mode, float & rcut_r,
                        int & max_cpy_trial, int & max_nnei_trial,
                        int *& array_int, unsigned long long *& array_longlong, bool & b_nlist_map,
                        deepmd::InputNlist &gpu_inlist, std::vector<int> & sec_a)
{
    FPTYPE * p_em = descrpt_tensor.data_ptr<FPTYPE>();
    FPTYPE * p_em_deriv = descrpt_deriv_tensor.data_ptr<FPTYPE>();
    FPTYPE * p_rij = rij_tensor.data_ptr<FPTYPE>();
    int * p_nlist = nlist_tensor.data_ptr<int>();
    const FPTYPE * p_coord = coord_tensor.data_ptr<FPTYPE>();
    const FPTYPE * p_box = box_tensor.data_ptr<FPTYPE>();
    const FPTYPE * avg = avg_tensor.data_ptr<FPTYPE>();
    const FPTYPE * std = std_tensor.data_ptr<FPTYPE>();
    const int * p_type = type_tensor.data_ptr<int>();

    // loop over samples
    for(int64_t ff = 0; ff < nsamples; ++ff){
        FPTYPE * em = p_em + ff*nloc*ndescrpt;
        FPTYPE * em_deriv = p_em_deriv + ff*nloc*ndescrpt*3;
        FPTYPE * rij = p_rij + ff*nloc*nnei*3;
        int * nlist = p_nlist + ff*nloc*nnei;
        const FPTYPE * coord = p_coord + ff*nall*3;
        const FPTYPE * box = p_box + ff*9;
        const int * type = p_type + ff*nall;

        if(descrpt_tensor.device().type() == torch::kCUDA) {
            #if GOOGLE_CUDA
            int * idx_mapping = NULL;
            int * ilist = NULL, * numneigh = NULL;
            int ** firstneigh = NULL;
            deepmd::malloc_device_memory(firstneigh, nloc);
            int * jlist = NULL;
            FPTYPE * coord_cpy;
            int * type_cpy;
            int frame_nall = nall;
            int mesh_tensor_size = static_cast<int>(mesh_tensor.numel());
            std::vector<torch::Tensor> tensor_list(7);
            // prepare coord and nlist
            _prepare_coord_nlist_gpu<FPTYPE>(
                descrpt_tensor, &tensor_list[0], &coord, coord_cpy, &type, type_cpy, idx_mapping, 
                gpu_inlist, ilist, numneigh, firstneigh, jlist, nbor_list_dev,
                frame_nall, mem_cpy, mem_nnei, max_nbor_size,
                box, mesh_tensor.data_ptr<int>(), mesh_tensor_size, nloc, nei_mode, rcut_r, max_cpy_trial, max_nnei_trial);
            // allocate temp memory, temp memory must not be used after this operation!

            std::vector<int64_t> int_shape{int64_t(sec_a.size()) + int64_t(nloc) * int64_t(sec_a.size()) + int64_t(nloc)};
            auto int_shape_option = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided)
                                                .device(descrpt_tensor.device()).requires_grad(false);
            torch::Tensor int_temp = torch::empty(int_shape, int_shape_option);

            std::vector<int64_t> uint64_shape{int64_t(nloc) * max_nbor_size * 2};
            //warning: original data type is uint64
            auto uint64_shape_option = torch::TensorOptions().dtype(torch::kInt64).layout(torch::kStrided)
                                                .device(descrpt_tensor.device()).requires_grad(false);
            torch::Tensor uint64_temp = torch::empty(uint64_shape, uint64_shape_option);
            array_int = int_temp.data_ptr<int>();
            array_longlong = (unsigned long long*)uint64_temp.data_ptr();

            // launch the gpu(nv) compute function
            deepmd::prod_env_mat_a_gpu_cuda(
                em, em_deriv, rij, nlist,
                coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a);
            if(b_nlist_map) _map_nlist_gpu(nlist, idx_mapping, nloc, nnei);
            deepmd::delete_device_memory(firstneigh);
            #endif //GOOGLE_CUDA

            #if TENSORFLOW_USE_ROCM
            int * idx_mapping = NULL;
            int * ilist = NULL, * numneigh = NULL;
            int ** firstneigh = NULL;
            deepmd::malloc_device_memory(firstneigh, nloc);
            int * jlist = NULL;
            FPTYPE * coord_cpy;
            int * type_cpy;
            int frame_nall = nall;
            int mesh_tensor_size = static_cast<int>(mesh_tensor.NumElements());
            std::vector<Tensor> tensor_list(7);
            // prepare coord and nlist
            _prepare_coord_nlist_gpu_rocm<FPTYPE>(
                context, &tensor_list[0], &coord, coord_cpy, &type, type_cpy, idx_mapping, 
                gpu_inlist, ilist, numneigh, firstneigh, jlist, nbor_list_dev,
                frame_nall, mem_cpy, mem_nnei, max_nbor_size,
                box, mesh_tensor.flat<int>().data(), mesh_tensor_size, nloc, nei_mode, rcut_r, max_cpy_trial, max_nnei_trial);

            // allocate temp memory, temp memory must not be used after this operation!
            Tensor int_temp;
            TensorShape int_shape;
            int_shape.AddDim(sec_a.size() + int_64(nloc) * sec_a.size() + nloc);
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, int_shape, &int_temp));
            Tensor uint64_temp;
            TensorShape uint64_shape;
            uint64_shape.AddDim(int_64(nloc) * max_nbor_size * 2);
            OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape, &uint64_temp));
            array_int = int_temp.flat<int>().data(); 
            array_longlong = uint64_temp.flat<unsigned long long>().data();

            // launch the gpu(nv) compute function
            deepmd::prod_env_mat_a_gpu_rocm(
                em, em_deriv, rij, nlist, 
                coord, type, gpu_inlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a);
            if(b_nlist_map) _map_nlist_gpu_rocm(nlist, idx_mapping, nloc, nnei);
            deepmd::delete_device_memory(firstneigh);
            #endif //TENSORFLOW_USE_ROCM
        }
        else if (descrpt_tensor.device().type() == torch::kCPU) {
            deepmd::InputNlist inlist;
            // some buffers, be freed after the evaluation of this frame
            std::vector<int> idx_mapping;
            std::vector<int> ilist(nloc), numneigh(nloc);
            std::vector<int*> firstneigh(nloc);
            std::vector<std::vector<int>> jlist(nloc);
            std::vector<FPTYPE> coord_cpy;
            std::vector<int> type_cpy;
            int frame_nall = nall;
            // prepare coord and nlist
            _prepare_coord_nlist_cpu<FPTYPE>(
                descrpt_tensor, &coord, coord_cpy, &type, type_cpy, idx_mapping, 
                inlist, ilist, numneigh, firstneigh, jlist,
                frame_nall, mem_cpy, mem_nnei, max_nbor_size,
                box, mesh_tensor.data_ptr<int>(), nloc, nei_mode, rcut_r, max_cpy_trial, max_nnei_trial);
            // launch the cpu compute function
            deepmd::prod_env_mat_a_cpu(
                em, em_deriv, rij, nlist, 
                coord, type, inlist, max_nbor_size, avg, std, nloc, frame_nall, rcut_r, rcut_r_smth, sec_a);
            // do nlist mapping if coords were copied
            if(b_nlist_map) _map_nlist_cpu(nlist, &idx_mapping[0], nloc, nnei);
        }
    }

}

template<typename FPTYPE>
static int
_norm_copy_coord_cpu(
    std::vector<FPTYPE> & coord_cpy,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r)
{
  std::vector<FPTYPE> tmp_coord(nall*3);
  std::copy(coord, coord+nall*3, tmp_coord.begin());
  deepmd::Region<FPTYPE> region;
  init_region_cpu(region, box);
  normalize_coord_cpu(&tmp_coord[0], nall, region);
  int tt;
  for(tt = 0; tt < max_cpy_trial; ++tt){
    coord_cpy.resize(mem_cpy*3);
    type_cpy.resize(mem_cpy);
    idx_mapping.resize(mem_cpy);
    int ret = copy_coord_cpu(
	&coord_cpy[0], &type_cpy[0], &idx_mapping[0], &nall, 
	&tmp_coord[0], type, nloc, mem_cpy, rcut_r, region);
    if(ret == 0){
      break;
    }
    else{
      mem_cpy *= 2;
    }
  }
  return (tt != max_cpy_trial);
}

template<typename FPTYPE>
static int
_build_nlist_cpu(
    std::vector<int> &ilist, 
    std::vector<int> &numneigh,
    std::vector<int*> &firstneigh,
    std::vector<std::vector<int>> &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r)
{
  int tt;
  for(tt = 0; tt < max_nnei_trial; ++tt){
    for(int ii = 0; ii < nloc; ++ii){
      jlist[ii].resize(mem_nnei);
      firstneigh[ii] = &jlist[ii][0];
    }
    deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
    int ret = build_nlist_cpu(
	inlist, &max_nnei, 
	coord, nloc, new_nall, mem_nnei, rcut_r);
    if(ret == 0){
      break;
    }
    else{
      mem_nnei *= 2;
    }
  }
  return (tt != max_nnei_trial);
}
    
static void
_map_nlist_cpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei)
{
  for (int64_t ii = 0; ii < nloc; ++ii){
    for (int64_t jj = 0; jj < nnei; ++jj){
      int record = nlist[ii*nnei+jj];
      if (record >= 0) {		
	nlist[ii*nnei+jj] = idx_mapping[record];	      
      }
    }
  }  
}

template <typename FPTYPE>
static void
_prepare_coord_nlist_cpu(
    torch::Tensor & descrpt_tensor,
    FPTYPE const ** coord,
    std::vector<FPTYPE> & coord_cpy,
    int const** type,
    std::vector<int> & type_cpy,
    std::vector<int> & idx_mapping,
    deepmd::InputNlist & inlist,
    std::vector<int> & ilist,
    std::vector<int> & numneigh,
    std::vector<int*> & firstneigh,
    std::vector<std::vector<int>> & jlist,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial)
{    
    inlist.inum = nloc;
    if(nei_mode != 3){
        // build nlist by myself
        // normalize and copy coord
        if(nei_mode == 1){
            int copy_ok = _norm_copy_coord_cpu(
                coord_cpy, type_cpy, idx_mapping, new_nall, mem_cpy,
                *coord, box, *type, nloc, max_cpy_trial, rcut_r);
            TORCH_CHECK(copy_ok, "cannot allocate mem for copied coords");
            *coord = &coord_cpy[0];
            *type = &type_cpy[0];
        }
        // build nlist
        int build_ok = _build_nlist_cpu(
            ilist, numneigh, firstneigh, jlist, max_nbor_size, mem_nnei,
            *coord, nloc, new_nall, max_nnei_trial, rcut_r);
        TORCH_CHECK(build_ok, "cannot allocate mem for nlist");
        inlist.ilist = &ilist[0];
        inlist.numneigh = &numneigh[0];
        inlist.firstneigh = &firstneigh[0];
    }
    else{
        // copy pointers to nlist data
        memcpy(&inlist.ilist, 4 + mesh_tensor_data, sizeof(int *));
        memcpy(&inlist.numneigh, 8 + mesh_tensor_data, sizeof(int *));
        memcpy(&inlist.firstneigh, 12 + mesh_tensor_data, sizeof(int **));
        max_nbor_size = deepmd::max_numneigh(inlist);
    }
}

#if GOOGLE_CUDA
template<typename FPTYPE>
static int
_norm_copy_coord_gpu(
    torch::Tensor & descrpt_tensor,
    torch::Tensor * tensor_list,
    FPTYPE * & coord_cpy,
    int * & type_cpy,
    int * & idx_mapping,
    int & nall,
    int & mem_cpy,
    const FPTYPE * coord,
    const FPTYPE * box,
    const int * type,
    const int &nloc, 
    const int &max_cpy_trial, 
    const float & rcut_r)
{
  // Tensor FPTYPE_temp;
    std::vector<int64_t> FPTYPE_shape{nall*3};
    auto FPTYPE_shape_option = torch::TensorOptions().dtype(descrpt_tensor.dtype()).layout(torch::kStrided)
                                                    .device(descrpt_tensor.device()).requires_grad(false);
    *tensor_list = torch::empty(FPTYPE_shape, FPTYPE_shape_option);

    FPTYPE * tmp_coord = tensor_list->data_ptr<FPTYPE>();
    DPErrcheck(cudaMemcpy(tmp_coord, coord, sizeof(FPTYPE) * nall * 3, cudaMemcpyDeviceToDevice));
    
    deepmd::Region<FPTYPE> region;
    init_region_cpu(region, box);
    FPTYPE box_info[18];
    std::copy(region.boxt, region.boxt+9, box_info);
    std::copy(region.rec_boxt, region.rec_boxt+9, box_info+9);
    int cell_info[23];
    deepmd::compute_cell_info(cell_info, rcut_r, region);
    const int loc_cellnum=cell_info[21];
    const int total_cellnum=cell_info[22];
    //Tensor double_temp;
    std::vector<int64_t> double_shape{18};
    *(tensor_list+1) = torch::empty(double_shape, FPTYPE_shape_option);

    //Tensor int_temp;
    std::vector<int64_t> int_shape{23+nloc*3+loc_cellnum+total_cellnum*3+total_cellnum*3+loc_cellnum+1+total_cellnum+1+nloc};
    auto int_shape_option = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided)
                                                .device(descrpt_tensor.device()).requires_grad(false);
    *(tensor_list+2) = torch::empty(int_shape, int_shape_option);

    FPTYPE * box_info_dev = (*(tensor_list+1)).data_ptr<FPTYPE>();
    FPTYPE * box_info_dev1 = (tensor_list+1)->data_ptr<FPTYPE>();

    torch::Tensor tmp = torch::empty(double_shape, torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided)
                                                .device(descrpt_tensor.device()).requires_grad(false));

    int * cell_info_dev = (*(tensor_list+2)).data_ptr<int>();
    int * int_data_dev = cell_info_dev + 23;

    deepmd::memcpy_host_to_device(box_info_dev, box_info, 18);
    deepmd::memcpy_host_to_device(cell_info_dev, cell_info, 23);
    deepmd::Region<FPTYPE> region_dev;
    FPTYPE * new_boxt = region_dev.boxt;
    FPTYPE * new_rec_boxt = region_dev.rec_boxt;
    region_dev.boxt = box_info_dev;
    region_dev.rec_boxt = box_info_dev + 9;
    
    deepmd::normalize_coord_gpu(tmp_coord, nall, region_dev);
    int tt;
    for(tt = 0; tt < max_cpy_trial; ++tt){
        //Tensor cpy_temp;
        std::vector<int64_t> cpy_shape{mem_cpy*3};
        *(tensor_list+3) = torch::empty(cpy_shape, FPTYPE_shape_option);

        //Tensor t_temp;
        std::vector<int64_t> t_shape(mem_cpy*2);
        auto t_shape_option = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided)
                                            .device(descrpt_tensor.device()).requires_grad(false);
        *(tensor_list+4) = torch::empty(cpy_shape, t_shape_option);

        coord_cpy = (*(tensor_list+3)).data_ptr<FPTYPE>();
        type_cpy = (*(tensor_list+4)).data_ptr<int>();
        idx_mapping = type_cpy + mem_cpy;
        int ret = deepmd::copy_coord_gpu(
            coord_cpy, type_cpy, idx_mapping, &nall, int_data_dev,
            tmp_coord, type, nloc, mem_cpy, loc_cellnum, total_cellnum, cell_info_dev, region_dev);
        if(ret == 0){
            break;
        }
        else{
            mem_cpy *= 2;
        }
    }
    region_dev.boxt = new_boxt;
    region_dev.rec_boxt = new_rec_boxt;
    return (tt != max_cpy_trial);
}

template<typename FPTYPE>
static int
_build_nlist_gpu(
    torch::Tensor & descrpt_tensor,
    torch::Tensor * tensor_list,
    int * &ilist, 
    int * &numneigh,
    int ** &firstneigh,
    int * &jlist,
    int & max_nnei,
    int & mem_nnei,
    const FPTYPE *coord,
    const int & nloc,
    const int & new_nall,
    const int & max_nnei_trial,
    const float & rcut_r)
{
    //Tensor nlist_temp;
    std::vector<int64_t> nlist_shape{nloc*2};
    auto nlist_shape_option = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided)
                                                  .device(descrpt_tensor.device()).requires_grad(false);
    *tensor_list = torch::empty(nlist_shape, nlist_shape_option);

    ilist = (*tensor_list).data_ptr<int>();
    numneigh = ilist + nloc;
    //Tensor jlist_temp;
    int * ind_data = NULL;
    
    std::vector<int*> firstneigh_host(nloc);
    int tt;
    for(tt = 0; tt < max_nnei_trial; ++tt){
        std::vector<int64_t> jlist_shape{3*int64_t(nloc)*mem_nnei};
        *(tensor_list+1) = torch::empty(jlist_shape, nlist_shape_option);
        jlist = (*(tensor_list+1)).data_ptr<int>();
        ind_data = jlist + nloc * mem_nnei;
        for(int64_t ii = 0; ii < nloc; ++ii){
            firstneigh_host[ii] = jlist + ii * mem_nnei;
        }
        deepmd::memcpy_host_to_device(firstneigh, firstneigh_host);
        deepmd::InputNlist inlist(nloc, ilist, numneigh, firstneigh);
        int ret = deepmd::build_nlist_gpu(
            inlist, &max_nnei, ind_data, 
            coord, nloc, new_nall, mem_nnei, rcut_r);
        if(ret == 0){
            break;
        }
        else{
            mem_nnei *= 2;
        }
    }
    return (tt != max_nnei_trial);
}

static void
_map_nlist_gpu(
    int * nlist,
    const int * idx_mapping,
    const int & nloc,
    const int & nnei)
{
  deepmd::use_nlist_map(nlist, idx_mapping, nloc, nnei);
}

template <typename FPTYPE>
static void
_prepare_coord_nlist_gpu(
    torch::Tensor & descrpt_tensor,
    torch::Tensor * tensor_list,
    FPTYPE const ** coord,
    FPTYPE * & coord_cpy,
    int const** type,
    int * & type_cpy,
    int * & idx_mapping,
    deepmd::InputNlist & inlist,
    int * & ilist,
    int * & numneigh,
    int ** & firstneigh,
    int * & jlist,
    int * & nbor_list_dev,
    int & new_nall,
    int & mem_cpy,
    int & mem_nnei,
    int & max_nbor_size,
    const FPTYPE * box,
    const int * mesh_tensor_data,
    const int mesh_tensor_size,
    const int & nloc,
    const int & nei_mode,
    const float & rcut_r,
    const int & max_cpy_trial,
    const int & max_nnei_trial)
{    
  if(nei_mode != 3){
    inlist.inum = nloc;
    // build nlist by myself
    // normalize and copy coord
    if(nei_mode == 1){
        int copy_ok = _norm_copy_coord_gpu(
            descrpt_tensor, tensor_list, coord_cpy, type_cpy, idx_mapping, new_nall, mem_cpy,
            *coord, box, *type, nloc, max_cpy_trial, rcut_r);
        TORCH_CHECK(copy_ok, "copy_ok error!");
        *coord = coord_cpy;
        *type = type_cpy;
    }
    //build nlist
    int build_ok = _build_nlist_gpu(
        descrpt_tensor, tensor_list + 5, ilist, numneigh, firstneigh, jlist, max_nbor_size, mem_nnei,
        *coord, nloc, new_nall, max_nnei_trial, rcut_r);
    TORCH_CHECK(build_ok, "");
    if (max_nbor_size <= 1024) {
        max_nbor_size = 1024;
    }
    else if (max_nbor_size <= 2048) {
        max_nbor_size = 2048;
    }
    else {
        max_nbor_size = 4096;
    }
    inlist.ilist = ilist;
    inlist.numneigh = numneigh;
    inlist.firstneigh = firstneigh;
  }
  else{
    // update nbor list
    deepmd::InputNlist inlist_temp;
    inlist_temp.inum = nloc;
    deepmd::env_mat_nbor_update(
        inlist_temp, inlist, max_nbor_size, nbor_list_dev,
        mesh_tensor_data, mesh_tensor_size);
    TORCH_CHECK((deepmd::max_numneigh(inlist_temp) <= max_nbor_size), "Assert failed, max neighbor size of atom(lammps) %d is larger than %d, which currently is not supported by deepmd-kit.", deepmd::max_numneigh(inlist_temp), max_nbor_size);
  }
}
#endif  // GOOGLE_CUDA

std::vector<torch::Tensor> prod_env_mat_a(torch::Tensor coord_tensor, torch::Tensor type_tensor,
                                            torch::Tensor natoms_tensor, torch::Tensor box_tensor,
                                            torch::Tensor mesh_tensor,
                                            torch::Tensor avg_tensor, torch::Tensor std_tensor,
                                            double rcut_a_d, double rcut_r_d, double rcut_r_smth_d,
                                            torch::Tensor sel_a_tensor, torch::Tensor sel_r_tensor)
{
    TORCH_CHECK(coord_tensor.sizes().size() == 2, "Dim of coord should be 2");
    TORCH_CHECK(type_tensor.sizes().size() == 2, "Dim of type should be 2");
    TORCH_CHECK(natoms_tensor.sizes().size() == 1, "Dim of natoms should be 1");
    TORCH_CHECK(box_tensor.sizes().size() == 2, "Dim of box should be 2");
    TORCH_CHECK(mesh_tensor.sizes().size() == 1, "Dim of mesh should be 1");
    TORCH_CHECK(avg_tensor.sizes().size() == 2, "Dim of avg should be 2");
    TORCH_CHECK(std_tensor.sizes().size() == 2, "Dim of std should be 2");
    TORCH_CHECK(natoms_tensor.size(0) >= 3, "number of atoms should be larger than (or equal to) 3");
    TORCH_CHECK((natoms_tensor.dtype() == torch::kInt32), "Type of natoms should be int32");
    TORCH_CHECK(natoms_tensor.device().type() == torch::kCPU, "natoms must on cpu");
    TORCH_CHECK(box_tensor.device().type() == torch::kCPU, "box must on cpu");

    int ndescrpt, ndescrpt_a, ndescrpt_r;
    int nnei, nnei_a, nnei_r, max_nbor_size;
    int mem_cpy, max_cpy_trial;
    int mem_nnei, max_nnei_trial;
    int *array_int = NULL;
    unsigned long long *array_longlong = NULL;
    deepmd::InputNlist gpu_inlist;
    int *nbor_list_dev = NULL;

    float rcut_a = static_cast<float>(rcut_a_d);
    float rcut_r = static_cast<float>(rcut_r_d);
    float rcut_r_smth = static_cast<float>(rcut_r_smth_d);

    std::vector<int> sec_a;
    std::vector<int> sec_r;

    TORCH_CHECK(sel_a_tensor.sizes().size() == 1 && sel_a_tensor.dtype() == torch::kInt32, "sel_a shape error!");
    TORCH_CHECK(sel_r_tensor.sizes().size() == 1 && sel_r_tensor.dtype() == torch::kInt32, "sel_r shape error!");
    std::vector<int> sel_a(sel_a_tensor.data_ptr<int>(), sel_a_tensor.data_ptr<int>() + sel_a_tensor.size(0));
    std::vector<int> sel_r(sel_r_tensor.data_ptr<int>(), sel_r_tensor.data_ptr<int>() + sel_r_tensor.size(0));

    deepmd::cum_sum(sec_a, sel_a);
    deepmd::cum_sum(sec_r, sel_r);
    ndescrpt_a = sec_a.back() * 4;
    ndescrpt_r = sec_r.back() * 1;
    ndescrpt = ndescrpt_a + ndescrpt_r;
    nnei_a = sec_a.back();
    nnei_r = sec_r.back();
    nnei = nnei_a + nnei_r;
    max_nbor_size = 1024;
    max_cpy_trial = 100;
    mem_cpy = 256;
    max_nnei_trial = 100;
    mem_nnei = 256;

    TORCH_CHECK(sec_r.back() == 0, "sec_r error!");

    auto natoms = natoms_tensor.flatten();
    int nloc = natoms[0].item<int>();
    int nall = natoms[1].item<int>();
    int ntypes = natoms_tensor.size(0) - 2; //nloc and nall mean something.
    int nsamples = coord_tensor.size(0);

    TORCH_CHECK(nsamples == type_tensor.size(0), "number of samples should match");
    TORCH_CHECK(nsamples == box_tensor.size(0), "number of samples should match");
    TORCH_CHECK(ntypes == avg_tensor.size(0), "number of avg should be ntype");
    TORCH_CHECK(ntypes == std_tensor.size(0), "number of std should be ntype");

    TORCH_CHECK(nall * 3 == coord_tensor.size(1), "number of atoms should match");
    TORCH_CHECK(nall == type_tensor.size(1), "number of atoms should match");
    TORCH_CHECK(9 == box_tensor.size(1), "number of box should be 9");
    TORCH_CHECK(ndescrpt == avg_tensor.size(1), "number of avg should be ndescrpt");
    TORCH_CHECK(ndescrpt == std_tensor.size(1), "number of std should be ndescrpt");

    TORCH_CHECK(ntypes == sel_a.size(), "number of types should match the length of sel array");
    TORCH_CHECK(ntypes == sel_r.size(), "number of types should match the length of sel array");

    int nei_mode = 0;
    bool b_nlist_map = false;
    if (mesh_tensor.size(0) == 16) {
      // lammps neighbor list
      nei_mode = 3;
    }
    else if (mesh_tensor.size(0) == 6) {
      // manual copied pbc
      assert (nloc == nall);
      nei_mode = 1;
      b_nlist_map = true;
    }
    else if (mesh_tensor.size(0) == 0) {
      // no pbc
      assert (nloc == nall);
      nei_mode = -1;
    }
    else {
      throw deepmd::deepmd_exception("invalid mesh tensor");
    }

    // Create output tensors
    std::vector<int64_t> descrpt_shape{nsamples, int64_t(nloc) * ndescrpt};
    std::vector<int64_t> descrpt_deriv_shape{nsamples, int64_t(nloc) * ndescrpt * 3};
    std::vector<int64_t> rij_shape{nsamples, int64_t(nloc) * nnei * 3};
    std::vector<int64_t> nlist_shape{nsamples, int64_t(nloc) * nnei};

    // define output tensor
    auto descrpt_tensor_option = torch::TensorOptions().dtype(coord_tensor.dtype()).layout(torch::kStrided)
                                                    .device(coord_tensor.device()).requires_grad(false);
    torch::Tensor descrpt_tensor = torch::empty(descrpt_shape, descrpt_tensor_option);
    torch::Tensor descrpt_deriv_tensor = torch::empty(descrpt_deriv_shape, descrpt_tensor_option);
    torch::Tensor rij_tensor = torch::empty(rij_shape, descrpt_tensor_option);
    auto nlist_tensor_option = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided)
                                                .device(coord_tensor.device()).requires_grad(false);
    torch::Tensor nlist_tensor = torch::empty(nlist_shape, nlist_tensor_option);
    if(coord_tensor.dtype() == torch::kFloat32){
        _prod_env_mat_a_generics<float>(descrpt_tensor, descrpt_deriv_tensor,
                                        rij_tensor, nlist_tensor, coord_tensor, type_tensor,
                                        natoms_tensor, box_tensor, mesh_tensor, avg_tensor, std_tensor,
                                        nsamples, nloc, ndescrpt, nnei, nall,
                                        nbor_list_dev, mem_cpy, mem_nnei, max_nbor_size,
                                        rcut_r_smth,
                                        nei_mode, rcut_r,
                                        max_cpy_trial, max_nnei_trial,
                                        array_int, array_longlong, b_nlist_map,
                                        gpu_inlist, sec_a);
    }else if(coord_tensor.dtype() == torch::kFloat64){
        //constexpr auto kFloat64 = at::kDouble
        _prod_env_mat_a_generics<double>(descrpt_tensor, descrpt_deriv_tensor,
                                         rij_tensor, nlist_tensor, coord_tensor, type_tensor,
                                         natoms_tensor, box_tensor, mesh_tensor, avg_tensor, std_tensor,
                                         nsamples, nloc, ndescrpt, nnei, nall,
                                         nbor_list_dev, mem_cpy, mem_nnei, max_nbor_size,
                                         rcut_r_smth,
                                         nei_mode, rcut_r,
                                         max_cpy_trial, max_nnei_trial,
                                         array_int, array_longlong, b_nlist_map,
                                         gpu_inlist, sec_a);
    }


    return {descrpt_tensor, descrpt_deriv_tensor, rij_tensor, nlist_tensor};
}


TORCH_LIBRARY(prod_env_mat, m){
    m.def("prod_env_mat_a", prod_env_mat_a);
}