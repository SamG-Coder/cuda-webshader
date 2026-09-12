// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include "division-probe.cu"
#define CHECK(x) do{if((x)!=cudaSuccess)return 2;}while(0)
int main(){float input[24],output[24],*a,*b;FILE*f=fopen("reports/division-input.bin","rb");if(!f||fread(input,4,24,f)!=24)return 2;fclose(f);CHECK(cudaMalloc(&a,96));CHECK(cudaMalloc(&b,96));CHECK(cudaMemcpy(a,input,96,cudaMemcpyHostToDevice));divisionProbe<<<1,16>>>(a,b);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output,b,96,cudaMemcpyDeviceToHost));f=fopen("reports/division-native.bin","wb");if(!f||fwrite(output,4,24,f)!=24)return 2;fclose(f);CHECK(cudaFree(a));CHECK(cudaFree(b));puts("PASS: 12 native scalar division and compound division cases captured.");}
