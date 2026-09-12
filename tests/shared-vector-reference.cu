// SPDX-License-Identifier: MIT
__device__ void blendReferences(float3& target,const float3& source){target+=source;}
__global__ void sharedInterpolate(float4* positions,float4* normals){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x,next=(lane+1)%128;
 __shared__ float3 sharedP[128];__shared__ float3 sharedN[128];float3 localP,localN;
 forwardInterpolation((float)i*.00390625f,sharedP[lane],sharedN[lane]);
 forwardInterpolation((float)i*.001953125f,localP,localN);
 blendReferences(localP,sharedP[lane]);
 __syncthreads();
 positions[i]=make_float4(sharedP[next]+localP,1.0f);normals[i]=make_float4(sharedN[next]+localN,0.0f);
}
