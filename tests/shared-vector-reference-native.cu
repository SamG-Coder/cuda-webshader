// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <helper_math.h>
#include <cstdio>
#include <vector>
#include "marching-interpolation.cuh"
#include "vector-reference.cu"
#include "shared-vector-reference.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){const int count=384;float4 *p,*n;CHECK(cudaMalloc(&p,count*16));CHECK(cudaMalloc(&n,count*16));sharedInterpolate<<<3,128>>>(p,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());std::vector<float4> data(count);for(int i=0;i<2;i++){CHECK(cudaMemcpy(data.data(),i?n:p,count*16,cudaMemcpyDeviceToHost));FILE* file=fopen(i?"reports/shared-vector-native-normals.bin":"reports/shared-vector-native-positions.bin","wb");if(!file)return 2;fwrite(data.data(),16,count,file);fclose(file);}CHECK(cudaFree(p));CHECK(cudaFree(n));printf("Captured 384 positions and gradients after shared interpolation, synchronization and neighbour reads; local and shared helper calls coexist.\n");return 0;}
