// Original project harness (MIT); included NVIDIA kernel retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "kernels/25.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int failures=0;for(auto shape:{std::pair<int,int>{1,32},{3,128},{17,256},{3907,256}}){int groups=shape.first,threads=shape.second;std::vector<int>data(26,-12345);for(int i=0;i<10;i++)data[i]=0;auto expected=data;for(int i=0;i<groups*threads&&i<NUM_THREADS;i++)expected[i%ARRAY_SIZE]++;int*g;CHECK(cudaMalloc(&g,data.size()*4));CHECK(cudaMemcpy(g,data.data(),data.size()*4,cudaMemcpyHostToDevice));cas_atomic<<<groups,threads>>>(g);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(data.data(),g,data.size()*4,cudaMemcpyDeviceToHost));bool pass=data==expected;printf("groups=%d threads=%d counters + guards: %s\n",groups,threads,pass?"PASS":"FAIL");failures+=!pass;CHECK(cudaFree(g));}return failures?1:0;}
