// Original project harness (MIT); included NVIDIA kernel retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <cstdio>
using uint = unsigned int; // Alias supplied by the upstream sample's helper headers.
#include "kernels/24.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int failures=0;for(auto shape:{std::pair<int,int>{1,32},{3,128},{4,256},{7,256}}){int blocks=shape.first,threads=shape.second,count=blocks*threads*4;unsigned values[]={0,1,16777217,2147483647,2147483648u,4294967294u,4294967295u},offsets[]={1,2147483648u,4294967295u,17};std::vector<unsigned>data(count+16,0xdeadbeef),buf(blocks);for(int i=0;i<count;i++)data[i]=values[i%7];for(int i=0;i<blocks;i++)buf[i]=offsets[i%4];auto expected=data;for(int i=0;i<count;i++)expected[i]+=buf[i/(threads*4)];unsigned *d,*b;CHECK(cudaMalloc(&d,data.size()*4));CHECK(cudaMalloc(&b,buf.size()*4));CHECK(cudaMemcpy(d,data.data(),data.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,buf.data(),buf.size()*4,cudaMemcpyHostToDevice));uniformUpdate<<<blocks,threads>>>((uint4*)d,b);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(data.data(),d,data.size()*4,cudaMemcpyDeviceToHost));bool pass=data==expected;printf("blocks=%d threads=%d uint4 output + guards: %s\n",blocks,threads,pass?"PASS":"FAIL");failures+=!pass;CHECK(cudaFree(d));CHECK(cudaFree(b));}return failures?1:0;}
