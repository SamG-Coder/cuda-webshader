// Original project regression harness, MIT.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "nbody-shared-memory.cuh"
#include "shared-wrapper.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(int threads:{32,128}){
 int n=threads*3;std::vector<float>input(n*4),output(n*4+16,-12345);for(int i=0;i<n*4;i++)input[i]=i*0.125f;
 float4 *a,*b;CHECK(cudaMalloc(&a,input.size()*4));CHECK(cudaMalloc(&b,output.size()*4));CHECK(cudaMemcpy(a,input.data(),input.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,output.data(),output.size()*4,cudaMemcpyHostToDevice));
 testSharedWrapper<<<3,threads,threads*16>>>(a,b);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),b,output.size()*4,cudaMemcpyDeviceToHost));
 for(int i=0;i<n;i++)for(int j=0;j<4;j++)if(output[i*4+j]!=input[((i/threads)*threads+threads-i%threads-1)*4+j])return 1;
 for(int i=n*4;i<n*4+16;i++)if(output[i]!=-12345)return 1;CHECK(cudaFree(a));CHECK(cudaFree(b));printf("threads=%d groups=3 PASS original shared wrapper, float4 tile reversal and guards\n",threads);
}return 0;}
