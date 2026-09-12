// Original project harness MIT; NVIDIA kernel and helper retain BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>
using uint=unsigned int;
#include "kernels/37.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(uint length:{256u,2048u})for(uint dir:{0u,1u}){uint n=length*2;std::vector<uint>input(n+16),values(n+16),zero(n+16,0xdeadbeef),keys(n+16),vals(n+16);uint special[]={0,0xffffffff,0x80000000,0x7fffffff,7,7,1,0};for(uint i=0;i<n+16;i++){input[i]=i<8?special[i]:i*1664525u+1013904223u;values[i]=0x80000000u+i;}uint *a,*b,*c,*d;CHECK(cudaMalloc(&a,(n+16)*4));CHECK(cudaMalloc(&b,(n+16)*4));CHECK(cudaMalloc(&c,(n+16)*4));CHECK(cudaMalloc(&d,(n+16)*4));CHECK(cudaMemcpy(a,input.data(),(n+16)*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,values.data(),(n+16)*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(c,zero.data(),(n+16)*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(d,zero.data(),(n+16)*4,cudaMemcpyHostToDevice));
 for(uint size=2;size<=length;size*=2)for(uint stride=size/2;stride;stride/=2){bitonicMergeGlobal<<<n/256,128>>>(c,d,a,b,length,size,stride,dir);CHECK(cudaGetLastError());std::swap(a,c);std::swap(b,d);}CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(keys.data(),a,(n+16)*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(vals.data(),b,(n+16)*4,cudaMemcpyDeviceToHost));
 for(uint batch=0;batch<2;batch++){uint start=batch*length;std::vector<uint>expected(input.begin()+start,input.begin()+start+length);std::sort(expected.begin(),expected.end());if(!dir)std::reverse(expected.begin(),expected.end());std::vector<bool>seen(length);for(uint i=0;i<length;i++){uint at=start+i,index=vals[at]-0x80000000u;if(keys[at]!=expected[i]||index<start||index>=start+length||input[index]!=keys[at]||seen[index-start]){printf("FAIL length=%u dir=%u index=%u\n",length,dir,i);return 1;}seen[index-start]=true;}}
 printf("length=%u batches=2 dir=%u PASS complete global-kernel sorting network, keys and value permutation\n",length,dir);CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));CHECK(cudaFree(d));}return 0;}
