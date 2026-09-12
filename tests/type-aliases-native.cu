// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
typedef unsigned int uint;
#include "type-aliases.cu"
int main(){uint* out;uint values[4]={};if(cudaMalloc(&out,16)!=cudaSuccess)return 2;aliasProbe<<<1,1>>>(out);auto e=cudaMemcpy(values,out,16,cudaMemcpyDeviceToHost);cudaFree(out);if(e!=cudaSuccess)return 2;uint expected[]={4,6,5,4};for(int i=0;i<4;i++)if(values[i]!=expected[i])return 1;printf("PASS aliases: byte truncation, scalar constructor, helper template and packed component: 4 6 5 4\n");}
