// SPDX-License-Identifier: MIT
__global__ void roundEven(float*input,float*output,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=rintf(input[i]);}
