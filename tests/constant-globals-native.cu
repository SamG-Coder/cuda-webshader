// Project regression harness MIT; included NVIDIA helper code is BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include "nbody-vector-traits.cuh"
#include "nbody-rsqrt.cuh"
#include "nbody-interaction.cuh"
#include "constant-globals.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(unsigned int n:{1u,129u,1025u})for(float softening:{0.0f,0.25f,2.0f}){std::vector<float>out((n+4)*4,-12345);float4*o;CHECK(cudaMalloc(&o,out.size()*4));CHECK(cudaMemcpy(o,out.data(),out.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpyToSymbol(softeningSquared,&softening,sizeof(float)));testBodyInteraction<<<(n+127)/128,128>>>(o,n,0);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(out.data(),o,out.size()*4,cudaMemcpyDeviceToHost));double maximum=0;for(unsigned int i=0;i<n;i++){double r[]={2-(i%8)*0.125,-1-(i%3)*0.25,0.5-(i%5)*0.125},s=1.5/std::pow(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+softening,1.5);for(int c=0;c<4;c++){double expected=c==3?1:r[c]*s,error=std::abs(out[i*4+c]-expected);maximum=std::fmax(maximum,error);if(!std::isfinite(out[i*4+c])||error>3e-6+2e-6*std::abs(expected))return 1;}}for(unsigned int i=n*4;i<out.size();i++)if(out[i]!=-12345)return 1;CHECK(cudaFree(o));printf("n=%u softening=%g PASS original N-body interaction and guards; maxError=%.9g\n",n,softening,maximum);}return 0;}
