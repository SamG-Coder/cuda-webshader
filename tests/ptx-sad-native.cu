#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "ptx-sad-kernel.cuh"
#define CHECK(x) do{if((x)!=cudaSuccess)return 2;}while(0)
int main(){const unsigned n=4100;std::vector<unsigned> input(n*3),out(n);unsigned seed=193;for(auto &v:input){seed=seed*1664525u+1013904223u;v=seed;}unsigned fixed[]={0,0xffffffffu,0,0x01020304u,0x04030201u,0xffffffffu,0xff000080u,0x008000ffu,123,0xffffffffu,0,0xffffffffu};for(int i=0;i<12;i++)input[i]=fixed[i];unsigned *a,*b;CHECK(cudaMalloc(&a,input.size()*4));CHECK(cudaMalloc(&b,n*4));CHECK(cudaMemcpy(a,input.data(),input.size()*4,cudaMemcpyHostToDevice));sadProbe<<<(n+127)/128,128>>>(a,b,n);CHECK(cudaMemcpy(out.data(),b,n*4,cudaMemcpyDeviceToHost));FILE*f=fopen("reports/ptx-sad-input.bin","wb");if(!f)return 2;fwrite(input.data(),4,input.size(),f);fclose(f);f=fopen("reports/ptx-sad-native.bin","wb");if(!f)return 2;fwrite(out.data(),4,n,f);fclose(f);CHECK(cudaFree(a));CHECK(cudaFree(b));printf("Captured %u original NVIDIA __usad4 results, including unsigned lanes and wrapping accumulators.\n",n);}
