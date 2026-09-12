// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
typedef unsigned int uint;
#include "linear-texture-kernel.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){const int length=32771,n=length+4;std::vector<unsigned char> bytes(length);std::vector<unsigned int> table(length),integers(n);std::vector<int> indices(n);std::vector<float> values(n);for(int i=0;i<length;i++){bytes[i]=(i*37+11)&255;table[i]=(unsigned(i)*2654435761u)^0xa5a5a5a5u;}for(int i=0;i<n;i++)indices[i]=i-2;
 unsigned char *b;unsigned int *t,*u;int* ix;float* f;CHECK(cudaMalloc(&b,length));CHECK(cudaMalloc(&t,length*4));CHECK(cudaMalloc(&u,n*4));CHECK(cudaMalloc(&ix,n*4));CHECK(cudaMalloc(&f,n*4));CHECK(cudaMemcpy(b,bytes.data(),length,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(t,table.data(),length*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ix,indices.data(),n*4,cudaMemcpyHostToDevice));
 cudaTextureObject_t textures[2];for(int i=0;i<2;i++){cudaResourceDesc r={};r.resType=cudaResourceTypeLinear;r.res.linear.devPtr=i?(void*)t:(void*)b;r.res.linear.sizeInBytes=length*(i?4:1);r.res.linear.desc=i?cudaCreateChannelDesc<unsigned int>():cudaCreateChannelDesc<unsigned char>();cudaTextureDesc d={};d.readMode=i?cudaReadModeElementType:cudaReadModeNormalizedFloat;CHECK(cudaCreateTextureObject(&textures[i],&r,&d,nullptr));}
 fetchLinear<<<(n+127)/128,128>>>(textures[0],textures[1],ix,f,u,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(values.data(),f,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(integers.data(),u,n*4,cudaMemcpyDeviceToHost));for(int i=0;i<2;i++){FILE* file=fopen(i?"reports/linear-texture-native-uint.bin":"reports/linear-texture-native-float.bin","wb");if(!file)return 2;fwrite(i?(void*)integers.data():(void*)values.data(),4,n,file);fclose(file);CHECK(cudaDestroyTextureObject(textures[i]));}CHECK(cudaFree(b));CHECK(cudaFree(t));CHECK(cudaFree(u));CHECK(cudaFree(ix));CHECK(cudaFree(f));printf("Captured %d float and %d uint results from %d linear records; indices -2 through %d.\n",n,n,length,length+1);return 0;}
