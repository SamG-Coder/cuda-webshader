// SPDX-License-Identifier: MIT
__device__ void forwardInterpolation(float t,float3& p,float3& n){vertexInterp2(t,make_float3(-1,2,3),make_float3(4,-3,8),make_float4(.5f,-1,2,0),make_float4(-2,3,.5f,1),p,n);}
__global__ void interpolate(float4* positions,float4* normals,unsigned int count){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float3 p[3],n[3];int slot=i%3;forwardInterpolation((float)i*.00390625f,p[slot],n[slot]);positions[i]=make_float4(p[slot],1.0f);normals[i]=make_float4(n[slot],0.0f);}}

