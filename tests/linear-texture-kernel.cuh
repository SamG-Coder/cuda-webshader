// SPDX-License-Identifier: MIT
__device__ float byteLookup(cudaTextureObject_t tex, int i) { return tex1Dfetch<float>(tex, i); }
__device__ unsigned int tableLookup(cudaTextureObject_t tex, int i) { return tex1Dfetch<uint>(tex, i); }
__global__ void fetchLinear(cudaTextureObject_t bytes, cudaTextureObject_t table, const int* indices, float* values, unsigned int* integers, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){values[i]=byteLookup(bytes,indices[i]);integers[i]=tableLookup(table,indices[i]);}
}
