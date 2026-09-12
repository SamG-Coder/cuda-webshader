// SPDX-License-Identifier: MIT
__constant__ float4 weights[125];
__device__ float sumWeight(uint i){float4 w=weights[i];return w.x+w.y+w.z+w.w;}
__global__ void vectorConstants(float* output){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<125)output[i]=sumWeight(i);}
