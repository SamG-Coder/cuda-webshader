// Original project harness (MIT); the included NVIDIA kernel retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "kernels/22.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
template<int TILE> int run(int M,int N,int K){
 std::vector<float>A(M*K),B(K*N),C(M*N+16,-12345);for(int i=0;i<M*K;i++)A[i]=(i%17-8)/16.f;for(int i=0;i<K*N;i++)B[i]=(i%13-6)/8.f;
 float *a,*b,*c;CHECK(cudaMalloc(&a,A.size()*4));CHECK(cudaMalloc(&b,B.size()*4));CHECK(cudaMalloc(&c,C.size()*4));CHECK(cudaMemcpy(a,A.data(),A.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,B.data(),B.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(c,C.data(),C.size()*4,cudaMemcpyHostToDevice));
 MatrixMulCUDA<TILE><<<dim3(N/TILE,M/TILE),dim3(TILE,TILE)>>>(c,a,b,K,N);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(C.data(),c,C.size()*4,cudaMemcpyDeviceToHost));
 int failures=0;for(int y=0;y<M;y++)for(int x=0;x<N;x++){double sum=0;for(int k=0;k<K;k++)sum+=double(A[y*K+k])*B[k*N+x];if(C[y*N+x]!=sum)failures++;}for(int i=M*N;i<M*N+16;i++)if(C[i]!=-12345)failures++;
 printf("tile=%d M=%d N=%d K=%d output + guards: %s\n",TILE,M,N,K,failures?"FAIL":"PASS");CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));return failures?1:0;
}
int main(){int errors=0;errors+=run<16>(16,16,16);errors+=run<16>(16,32,48);errors+=run<16>(48,16,32);errors+=run<16>(64,96,128);errors+=run<32>(32,32,32);errors+=run<32>(32,64,96);errors+=run<32>(96,32,64);errors+=run<32>(64,96,128);return errors?1:0;}
