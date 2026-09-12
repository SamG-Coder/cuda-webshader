// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
typedef unsigned int uint;
#include "vector-constants.cu"
int main(){float4 values[125];float actual[125];for(int i=0;i<125;i++)values[i]=make_float4(i*.125f,-i*.25f,3.5f,.25f);if(cudaMemcpyToSymbol(weights,values,sizeof(values))!=cudaSuccess)return 2;float* output;cudaMalloc(&output,sizeof(actual));vectorConstants<<<2,64>>>(output);auto e=cudaMemcpy(actual,output,sizeof(actual),cudaMemcpyDeviceToHost);cudaFree(output);if(e!=cudaSuccess)return 2;for(int i=0;i<125;i++)if(actual[i]!=3.75f-i*.125f)return 1;printf("PASS 125 float4 constants, 500 component uploads and helper reads\n");}
