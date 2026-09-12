// SPDX-License-Identifier: MIT
// Probe raw floating-point linear texture records and a vector alias/helper chain.
typedef float2 fComplex;
__device__ fComplex readPair(cudaTextureObject_t t,int index){return tex1Dfetch<fComplex>(t,index);}
__global__ void fetchFloatLinear(cudaTextureObject_t scalar,cudaTextureObject_t pair,cudaTextureObject_t quad,const int*indices,float*a,float2*b,float4*c,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int index=indices[i];a[i]=tex1Dfetch<float>(scalar,index);b[i]=readPair(pair,index);c[i]=tex1Dfetch<float4>(quad,index);}}
