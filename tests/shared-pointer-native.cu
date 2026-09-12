// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <helper_math.h>
#include <cstdio>
#include <vector>
#include "shared-pointer.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){float4* output;CHECK(cudaMalloc(&output,96*3*16));sharedPointers<<<3,32>>>(output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());std::vector<float4> data(96*3);CHECK(cudaMemcpy(data.data(),output,data.size()*16,cudaMemcpyDeviceToHost));for(int i=0;i<96;i++){float expected[]={0,0,2,0,float(i+1),float(i+1),0,1,float(i+1),0,2,1};for(int j=0;j<12;j++)if(reinterpret_cast<float*>(data.data())[i*12+j]!=expected[j])return 3;}FILE* f=fopen("reports/shared-pointer-native.bin","wb");if(!f)return 2;fwrite(data.data(),16,data.size(),f);fclose(f);CHECK(cudaFree(output));puts("PASS 1152 components: original calcNormal, aliased shared writes, pointer increments and forwarding.");}
