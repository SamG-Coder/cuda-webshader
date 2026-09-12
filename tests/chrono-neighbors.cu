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

// Unchanged kernels extracted from Chrono a92c6f72f422fbcafe0b37125d4070cb6a3b5803.
// Only the upstream uint alias and UINT_MAX macro are supplied below.
typedef unsigned int uint;
#define UINT_MAX 4294967295u
__global__ void OriginalToSortedD(uint* mapOriginalToSorted, uint* gridMarkerIndex, uint numActive) {
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numActive)
        return;

    uint index = gridMarkerIndex[id];

    mapOriginalToSorted[index] = id;
}

__global__ void findCellStartEndD(uint* cellStartD,        // output: cell start index
                                  uint* cellEndD,          // output: cell end index
                                  uint* gridMarkerHashD,   // input: sorted grid hashes
                                  uint* gridMarkerIndexD,  // input: sorted particle indices
                                  uint numActive) {
    extern __shared__ uint sharedHash[];  // blockSize + 1 elements
    // Get the particle index the current thread is supposed to be looking at.
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint hash;

    if (index >= numActive)
        return;

    hash = gridMarkerHashD[index];
    if (hash == UINT_MAX)
        return;
    // Load hash data into shared memory so that we can look at neighboring
    // particle's hash value without loading two hash values per thread
    sharedHash[threadIdx.x + 1] = hash;

    // first thread in block must load neighbor particle hash
    if (index > 0 && threadIdx.x == 0)
        sharedHash[0] = gridMarkerHashD[index - 1];

    __syncthreads();
    if (sharedHash[threadIdx.x] == UINT_MAX)
        return;

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell. As it
    // isn't the first particle, it must also be the cell end of
    // the previous particle's cell.
    if (index == 0 || hash != sharedHash[threadIdx.x]) {
        cellStartD[hash] = index;
        if (index > 0)
            cellEndD[sharedHash[threadIdx.x]] = index;
    }

    if (index == numActive - 1)
        cellEndD[hash] = index + 1;
}

