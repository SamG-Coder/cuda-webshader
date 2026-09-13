// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Arman Pazouki, Milad Rakhsha, Wei Hu, Radu Serban
// =============================================================================
//
// Base class for processing proximity in fsi system.
// =============================================================================

// Unchanged functions; types and parameter constants supplied by chrono-search.cu.
#include <cmath>
typedef int int32_t;
__host__ __device__ inline bool IsFinite(Real3 v) {
#ifdef __CUDA_ARCH__
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
#else
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
#endif
}
__global__ void reorderDataD(const uint* __restrict__ gridMarkerIndexD,
                             Real4* __restrict__ sortedPosRadD,
                             Real3* __restrict__ sortedVelMasD,
                             Real4* __restrict__ sortedRhoPreMuD,
                             Real3* __restrict__ sortedTauXxYyZzD,
                             Real3* __restrict__ sortedTauXyXzYzD,
                             Real3* __restrict__ sortedPcEvSvD,
                             int32_t* __restrict__ activityIdentifierSortedD,
                             const Real4* __restrict__ posRadD,
                             const Real3* __restrict__ velMasD,
                             const Real4* __restrict__ rhoPresMuD,
                             const Real3* __restrict__ tauXxYyZzD,
                             const Real3* __restrict__ tauXyXzYzD,
                             const Real3* __restrict__ pcEvSvD,
                             const int32_t* __restrict__ activityIdentifierOriginalD,
                             uint numActive) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numActive)
        return;

    uint originalIndex = gridMarkerIndexD[tid];

    // Read from original arrays
    Real4 posRadVal = posRadD[originalIndex];
    Real3 velMasVal = velMasD[originalIndex];
    Real4 rhoPreMuVal = rhoPresMuD[originalIndex];
    int32_t activityIdentifierVal = activityIdentifierOriginalD[originalIndex];

    if (!IsFinite(mR3(posRadVal))) {
        printf("Error! reorderDataD_ActiveOnly: posRad is NAN at original index %u\n", originalIndex);
    }

    // Write to sorted arrays at index 'tid'
    sortedPosRadD[tid] = posRadVal;
    sortedVelMasD[tid] = velMasVal;
    sortedRhoPreMuD[tid] = rhoPreMuVal;
    activityIdentifierSortedD[tid] = activityIdentifierVal;

    // For CRM only
    if (paramsD.physics_problem == PhysicsProblem::CRM) {
        Real3 tauXxYyZzVal = tauXxYyZzD[originalIndex];
        Real3 tauXyXzYzVal = tauXyXzYzD[originalIndex];
        Real3 pcEvSvVal = pcEvSvD[originalIndex];

        if (!IsFinite(tauXxYyZzVal)) {
            printf("Error! reorderDataD_ActiveOnly: tauXxYyZz is NAN at original index %u\n", originalIndex);
        }

        sortedTauXxYyZzD[tid] = tauXxYyZzVal;
        sortedTauXyXzYzD[tid] = tauXyXzYzVal;
        sortedPcEvSvD[tid] = pcEvSvVal;
    }
}

__global__ void OriginalToSortedD(uint* mapOriginalToSorted, uint* gridMarkerIndex, uint numActive) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    uint index = gridMarkerIndex[id];

    mapOriginalToSorted[index] = id;
}
