// Original project harness (MIT); included NVIDIA kernel retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <cstdio>
#include "kernels/30.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int failures=0;for(int log2N:{2,3,6,7,8,11}){int N=1<<log2N,batches=3;std::vector<float>input(N*batches),output(N*batches+16,-12345);for(int i=0;i<N*batches;i++)input[i]=(i%31-15)/16.f;auto expected=output;for(int b=0;b<batches;b++)for(int k=0;k<N;k++){double sum=0;for(int j=0;j<N;j++){int bits=j&k,parity=0;while(bits){parity^=1;bits&=bits-1;}sum+=(parity?-1:1)*input[b*N+j];}expected[b*N+k]=float(sum);}float *in,*out;CHECK(cudaMalloc(&in,input.size()*4));CHECK(cudaMalloc(&out,output.size()*4));CHECK(cudaMemcpy(in,input.data(),input.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(out,output.data(),output.size()*4,cudaMemcpyHostToDevice));fwtBatch1Kernel<<<batches,N/4,N*4>>>(out,in,log2N);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),out,output.size()*4,cudaMemcpyDeviceToHost));bool pass=output==expected;printf("N=%d batches=%d dynamicBytes=%d output + guards: %s\n",N,batches,N*4,pass?"PASS":"FAIL");failures+=!pass;CHECK(cudaFree(in));CHECK(cudaFree(out));}return failures?1:0;}
