// Original Chrono reorderDataD; BSD terms in THIRD_PARTY_NOTICES.md.
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
