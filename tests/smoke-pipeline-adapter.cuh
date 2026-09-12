// SPDX-License-Identifier: MIT
// Host thrust::sequence equivalent for the sandbox's GPU pipeline.
__global__ void smokeIndices(unsigned* indices,unsigned count){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count)indices[i]=i;
}
