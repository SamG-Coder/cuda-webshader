// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include "indexed-macros.cu"
int main(){unsigned int* out;unsigned int value=0;if(cudaMalloc(&out,4)!=cudaSuccess)return 1;indexedTile<<<1,1>>>(out);auto status=cudaMemcpy(&value,out,4,cudaMemcpyDeviceToHost);cudaFree(out);if(status!=cudaSuccess||value!=0xff0d0b07u)return 2;printf("PASS indexed shared uchar4 macro and definition branches: %08x\n",value);}
