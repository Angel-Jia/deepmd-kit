#include "c_api.h"

#include <vector>
#include "c_api_internal.h"
#include "common.h"
#include "DeepPot.h"

extern "C" {

DP_DeepPot::DP_DeepPot(deepmd::DeepPot& dp)
    : dp(dp) {}

DP_DeepPot* DP_NewDeepPot(const char* c_model) {
    std::string model(c_model);
    deepmd::DeepPot dp(model);
    DP_DeepPot* new_dp = new DP_DeepPot(dp);
    return new_dp;
}
} // extern "C"

template <typename VALUETYPE>
void DP_DeepPotCompute_variant (
    DP_DeepPot* dp,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    double* energy,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE* atomic_energy,
    VALUETYPE* atomic_virial
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    double e;
    std::vector<VALUETYPE> f, v, ae, av;

    dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_);
    // copy from C++ vectors to C arrays, if not NULL pointer
    if(energy) *energy = e;
    if(force) std::copy(f.begin(), f.end(), force);
    if(virial) std::copy(v.begin(), v.end(), virial);
    if(atomic_energy) std::copy(ae.begin(), ae.end(), atomic_energy);
    if(atomic_virial) std::copy(av.begin(), av.end(), atomic_virial);
}

template
void DP_DeepPotCompute_variant <double> (
    DP_DeepPot* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    );

template
void DP_DeepPotCompute_variant <float> (
    DP_DeepPot* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    );

extern "C" {

void DP_DeepPotCompute (
    DP_DeepPot* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    ) {
    DP_DeepPotCompute_variant<double>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputef (
    DP_DeepPot* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    ) {
    DP_DeepPotCompute_variant<float>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_ConvertPbtxtToPb(
    const char* c_pbtxt,
    const char* c_pb
    ) {
    std::string pbtxt(c_pbtxt);
    std::string pb(c_pb);
    deepmd::convert_pbtxt_to_pb(pbtxt, pb);
}

} // extern "C"