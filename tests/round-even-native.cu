// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "round-even-kernel.cuh"
#define CUDA(x) do{if((x)!=cudaSuccess)return 2;}while(0)
bool save(const char*p,const std::vector<float>&v){FILE*f=fopen(p,"wb");if(!f)return false;bool ok=fwrite(v.data(),4,v.size(),f)==v.size();fclose(f);return ok;}
int main(){std::vector<float> input;for(int i=-3072;i<=3072;i++)input.push_back(i*.25f);for(float v:{-0.f,-8388608.f,8388608.f,-16777216.f,16777216.f,-3.402823466e38f,3.402823466e38f})input.push_back(v);std::vector<float> output(input.size());float*a,*b;CUDA(cudaMalloc(&a,input.size()*4));CUDA(cudaMalloc(&b,input.size()*4));CUDA(cudaMemcpy(a,input.data(),input.size()*4,cudaMemcpyHostToDevice));roundEven<<<(input.size()+127)/128,128>>>(a,b,unsigned(input.size()));CUDA(cudaMemcpy(output.data(),b,output.size()*4,cudaMemcpyDeviceToHost));if(!save("reports/round-even-input.bin",input)||!save("reports/round-even-native.bin",output))return 2;CUDA(cudaFree(a));CUDA(cudaFree(b));printf("Captured %zu rintf values, including halfway ties, signed zero and large finite floats.\n",input.size());}
