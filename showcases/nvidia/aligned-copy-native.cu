// Original project harness (MIT); included NVIDIA kernel retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include "kernels/26.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
template<class T,class Scalar>int run(const char* name,int n){int width=sizeof(T)/sizeof(Scalar),count=(n?n:1)*width;std::vector<Scalar>input(count),output(count+16,Scalar(-12345));for(int i=0;i<count;i++){if constexpr(std::is_same_v<Scalar,float>){float values[]={0.f,-0.f,.125f,-31.5f,65536.25f,1e-20f};input[i]=values[i%6];}else if constexpr(std::is_unsigned_v<Scalar>){unsigned values[]={0,1,16777217,2147483648u,4294967295u};input[i]=values[i%5];}else{int values[]={0,-1,16777217,(-2147483647-1),2147483647};input[i]=values[i%5];}}auto expected=output;memcpy(expected.data(),input.data(),n*width*sizeof(Scalar));T *in,*out;CHECK(cudaMalloc(&in,input.size()*sizeof(Scalar)));CHECK(cudaMalloc(&out,output.size()*sizeof(Scalar)));CHECK(cudaMemcpy(in,input.data(),input.size()*sizeof(Scalar),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(out,output.data(),output.size()*sizeof(Scalar),cudaMemcpyHostToDevice));testKernel<T><<<2,128>>>(out,in,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),out,output.size()*sizeof(Scalar),cudaMemcpyDeviceToHost));bool pass=memcmp(output.data(),expected.data(),output.size()*sizeof(Scalar))==0;printf("%s records=%d bytes + guards: %s\n",name,n,pass?"PASS":"FAIL");CHECK(cudaFree(in));CHECK(cudaFree(out));return pass?0:1;}
int main(){int failures=0;for(int n:{0,1,129,1031}){failures+=run<int,int>("int",n);failures+=run<uint4,unsigned>("uint4",n);failures+=run<float4,float>("float4",n);}return failures?1:0;}
