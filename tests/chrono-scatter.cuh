// Original Project Chrono marker grouping and WCSPH copy-back kernel.
// Upstream revision a92c6f72f422fbcafe0b37125d4070cb6a3b5803.
// BSD-3-Clause; see THIRD_PARTY_NOTICES.md.
enum class MarkerGroup { FLUID, SOLID, BOUNDARY, NON_FLUID, NON_SOLID, NON_BOUNDARY, ALL };
__host__ __device__ inline bool IsInMarkerGroup(MarkerGroup group, Real code) {
    switch (group) {
        case MarkerGroup::ALL:
            return true;
        case MarkerGroup::FLUID:
            return code < -0.5;
        case MarkerGroup::SOLID:
            return code > 0.5;
        case MarkerGroup::BOUNDARY:
            return code < 0.5 && code > -0.5;
        case MarkerGroup::NON_FLUID:
            return code > -0.5;
        case MarkerGroup::NON_SOLID:
            return code < 0.5;
        case MarkerGroup::NON_BOUNDARY:
            return code > 0.5 || code < -0.5;
    }
    return false;
}
__global__ void CopySortedToOriginalWCSPH_D(MarkerGroup group,
                                            const Real4* sortedPosRad,
                                            const Real3* sortedVelMas,
                                            const Real4* sortedRhoPresMu,
                                            const Real3* sortedTauXxYyZz,
                                            const Real3* sortedTauXyXXzYz,
                                            const Real3* sortedPcEvSv,
                                            const Real4* derivVelRho,
                                            const uint numActive,
                                            Real4* posRadOriginal,
                                            Real3* velMasOriginal,
                                            Real4* rhoPresMuOriginal,
                                            Real3* tauXxYyZzOriginal,
                                            Real3* tauXyXzYzOriginal,
                                            Real3* pcEvSvOriginal,
                                            Real4* derivVelRhoOriginal,
                                            uint* gridMarkerIndex) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    Real type = sortedRhoPresMu[id].w;
    if (!IsInMarkerGroup(group, type))
        return;

    uint index = gridMarkerIndex[id];
    posRadOriginal[index] = sortedPosRad[id];
    velMasOriginal[index] = sortedVelMas[id];
    rhoPresMuOriginal[index] = sortedRhoPresMu[id];
    derivVelRhoOriginal[index] = derivVelRho[id];
    tauXxYyZzOriginal[index] = sortedTauXxYyZz[id];
    tauXyXzYzOriginal[index] = sortedTauXyXXzYz[id];
    pcEvSvOriginal[index] = sortedPcEvSv[id];
}
