// Original project harness (MIT); included NVIDIA file retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "kernels/36.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(int n:{0,1,127,128,129,257,1025}){int size=n+16;std::vector<float>A(size),B(size),C(size,-12345);for(int i=0;i<size;i++){A[i]=(i%37-18)/8.0f;B[i]=(i%19-9)/4.0f;}float *a,*b,*c;CHECK(cudaMalloc(&a,size*4));CHECK(cudaMalloc(&b,size*4));CHECK(cudaMalloc(&c,size*4));CHECK(cudaMemcpy(a,A.data(),size*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,B.data(),size*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(c,C.data(),size*4,cudaMemcpyHostToDevice));VecAdd_kernel<<<n?(n+127)/128:1,128>>>(a,b,c,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(C.data(),c,size*4,cudaMemcpyDeviceToHost));for(int i=0;i<size;i++)if(C[i]!=(i<n?A[i]+B[i]:-12345)){printf("n=%d FAIL at %d\n",n,i);return 1;}printf("n=%d PASS exact output and 16 guards\n",n);CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));}return 0;}
