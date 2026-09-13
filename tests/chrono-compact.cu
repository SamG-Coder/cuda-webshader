// Copyright (c) 2014 projectchrono.org. All rights reserved.
// Original Chrono declarations at a92c6f72f422fbcafe0b37125d4070cb6a3b5803.
// BSD license reproduced in THIRD_PARTY_NOTICES.md.
typedef unsigned int uint;
typedef int int32_t;
__global__ void fillActiveListD(const uint* __restrict__ prefixSum, const int32_t* __restrict__ extendedActivityIdD, uint* __restrict__ activeListD, uint numAllMarkers) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numAllMarkers)
        return;

    // Check if the value is 1 (active)
    if (extendedActivityIdD[tid] == 1) {
        uint writePos = prefixSum[tid];  // an integer in [0..(numActive-1)]
        activeListD[writePos] = tid;
    }
}

// MIT host-adapter kernels; no original Chrono kernel or functor is rewritten.
__global__ void normalizeActivity(const int32_t* activity,uint* positive,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)positive[i]=activity[i]>0?1u:0u;}
__global__ void gatherSelected(const float4* positions,const uint* activeList,float4* selected,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)selected[i]=positions[activeList[i]];}
