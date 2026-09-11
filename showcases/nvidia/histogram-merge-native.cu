// Original project harness (MIT); included NVIDIA kernels retain BSD-3-Clause.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <cstdio>
using uint=unsigned int;
#include "kernels/31.cu"
#include "kernels/32.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int failures=0;for(int bins:{64,256})for(int count:{0,1,17,255,256,513}){std::vector<unsigned>input((count?count:1)*bins),output(bins+16,0xdeadbeef);for(unsigned i=0;i<input.size();i++){unsigned values[]={0,1,16777217,2147483648u,4294967295u,i%101};input[i]=values[i%6];}auto expected=output;for(int bin=0;bin<bins;bin++){unsigned sum=0;for(int row=0;row<count;row++)sum+=input[row*bins+bin];expected[bin]=sum;}unsigned *in,*out;CHECK(cudaMalloc(&in,input.size()*4));CHECK(cudaMalloc(&out,output.size()*4));CHECK(cudaMemcpy(in,input.data(),input.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(out,output.data(),output.size()*4,cudaMemcpyHostToDevice));if(bins==64)mergeHistogram64Kernel<<<bins,256>>>(out,in,count);else mergeHistogram256Kernel<<<bins,256>>>(out,in,count);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),out,output.size()*4,cudaMemcpyDeviceToHost));bool pass=output==expected;printf("bins=%d partialHistograms=%d output + guards: %s\n",bins,count,pass?"PASS":"FAIL");failures+=!pass;CHECK(cudaFree(in));CHECK(cudaFree(out));}return failures?1:0;}
