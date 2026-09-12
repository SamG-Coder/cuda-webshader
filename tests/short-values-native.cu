// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include "short-values.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int *dst;CHECK(cudaMalloc(&dst,32));FILE* f=fopen("reports/short-values-native.bin","wb");if(!f)return 2;
int values[][2]={{32766,65534},{-32768,0},{-1,65535},{0,32768}};
for(auto &v:values){shortValues<<<1,1>>>((short)v[0],(unsigned short)v[1],dst);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());int out[8];CHECK(cudaMemcpy(out,dst,32,cudaMemcpyDeviceToHost));int expected[]={(short)(v[0]+4),(unsigned short)(v[1]+4),v[0]+v[1],2,2,abs(v[0]),-1,65535};for(int i=0;i<8;i++)if(out[i]!=expected[i])return 3;if(fwrite(out,4,8,f)!=8)return 2;printf("PASS short values a=%d b=%d\n",v[0],v[1]);}fclose(f);CHECK(cudaFree(dst));}
