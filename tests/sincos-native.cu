// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "sincos-kernel.cuh"
#define CUDA(x) do{if((x)!=cudaSuccess)return 2;}while(0)
bool save(const char*p,const void*d,size_t n){FILE*f=fopen(p,"wb");if(!f)return false;bool ok=fwrite(d,1,n,f)==n;fclose(f);return ok;}
int main(){const int n=8195;std::vector<float> phases(n);std::vector<int> bits(n),first(n);std::vector<float4> output(n);for(int i=0;i<n;i++){phases[i]=float(i-4096)*.0125f;bits[i]=i<32?int(1u<<i):i==32?0:int(unsigned(i)*2654435761u);}float*p;int*b,*f;float4*o;CUDA(cudaMalloc(&p,n*4));CUDA(cudaMalloc(&b,n*4));CUDA(cudaMalloc(&f,n*4));CUDA(cudaMalloc(&o,n*16));CUDA(cudaMemcpy(p,phases.data(),n*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(b,bits.data(),n*4,cudaMemcpyHostToDevice));sincosProbe<<<(n+127)/128,128>>>(p,b,o,f,n);CUDA(cudaMemcpy(output.data(),o,n*16,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(first.data(),f,n*4,cudaMemcpyDeviceToHost));if(!save("reports/sincos-phases.bin",phases.data(),n*4)||!save("reports/sincos-bits.bin",bits.data(),n*4)||!save("reports/sincos-output.bin",output.data(),n*16)||!save("reports/sincos-first.bin",first.data(),n*4))return 2;for(void*v:{(void*)p,(void*)b,(void*)f,(void*)o})CUDA(cudaFree(v));printf("Captured %d sincosf/__sincosf pairs and __ffs values, including all 32 one-bit positions and zero.\n",n);}
