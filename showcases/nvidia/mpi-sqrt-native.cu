// Original project harness (MIT); included NVIDIA kernel retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include "kernels/34.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int failures=0;for(auto shape:{std::pair<int,int>{1,1},{1,32},{2,128},{3,256}}){int groups=shape.first,threads=shape.second,n=groups*threads;std::vector<float>input(n),output(n+16,-12345);float values[]={0,.25f,1,2,4,16,.000001f,1e-30f,1.1754943508222875e-38f,3.4028234663852886e38f,16777216,.5f};for(int i=0;i<n;i++)input[i]=i<12?values[i]:float(double(i%997)/7);float*in,*out;CHECK(cudaMalloc(&in,input.size()*4));CHECK(cudaMalloc(&out,output.size()*4));CHECK(cudaMemcpy(in,input.data(),input.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(out,output.data(),output.size()*4,cudaMemcpyHostToDevice));simpleMPIKernel<<<groups,threads>>>(in,out);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),out,output.size()*4,cudaMemcpyDeviceToHost));bool pass=true;double maximum=0;for(int i=0;i<n+16;i++){double expected=i<n?sqrt(double(input[i])):-12345,error=fabs(output[i]-expected);if(i<n&&expected)maximum=fmax(maximum,error/expected);if(!std::isfinite(output[i])||error>2e-7*fabs(expected))pass=false;}printf("n=%d output + guards: %s maxRelativeError=%.9g\n",n,pass?"PASS":"FAIL",maximum);failures+=!pass;CHECK(cudaFree(in));CHECK(cudaFree(out));}return failures?1:0;}
