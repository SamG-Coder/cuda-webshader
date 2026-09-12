// SPDX-License-Identifier: MIT
__global__ void divisionProbe(const float*input,float*output){unsigned i=threadIdx.x;if(i<12){float a=input[2*i],b=input[2*i+1];output[2*i]=a/b;a/=b;output[2*i+1]=a;}}
