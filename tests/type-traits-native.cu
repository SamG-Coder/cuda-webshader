// Original regression harness, MIT; included NVIDIA vector traits are BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "nbody-vector-traits.cuh"
#include "type-traits.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(unsigned int n:{1u,129u,1025u}){std::vector<float>data((n+4)*4),output(data.size());for(unsigned int i=0;i<data.size();i++)data[i]=i<n*4?(int(i%17)-8)*0.125f:-12345;float4*positions;CHECK(cudaMalloc(&positions,data.size()*4));CHECK(cudaMemcpy(positions,data.data(),data.size()*4,cudaMemcpyHostToDevice));traitPositions<float><<<(n+127)/128,128>>>(positions,n,0.25f);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),positions,data.size()*4,cudaMemcpyDeviceToHost));float delta[]={0.125f,-0.0625f,0.25f,0};for(unsigned int i=0;i<data.size();i++)if(output[i]!=data[i]+(i<n*4?delta[i%4]:0))return 1;CHECK(cudaFree(positions));printf("n=%u PASS original NVIDIA vec3/vec4 traits, position update and guards\n",n);}return 0;}
