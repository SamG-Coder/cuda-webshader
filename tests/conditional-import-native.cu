// SPDX-License-Identifier: MIT
#include <cstdio>
#include "conditional-import.cu"
int main(){int* out;int value=0;if(cudaMalloc(&out,4)!=cudaSuccess)return 2;conditionalKernel<<<1,1>>>(out);auto e=cudaMemcpy(&value,out,4,cudaMemcpyDeviceToHost);cudaFree(out);if(e!=cudaSuccess||value!=(PICK?19:7))return 1;printf("PASS original conditional source PICK=%d value=%d\n",PICK,value);}
