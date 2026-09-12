// Original project harness (MIT); included NVIDIA kernel/helper retain BSD-3-Clause.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include "kernels/33.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
double reference(unsigned x){bool upper=x>=0x80000000u;unsigned low=upper?0xffffffffu-x:x;double p=(double(low)+.5)/4294967295.,a=0,b=10;for(int i=0;i<50;i++){double mid=(a+b)/2;if(.5*erfc(mid/sqrt(2.))>p)a=mid;else b=mid;}return (upper?1:-1)*(a+b)/2;}
int main(){int failures=0;unsigned special[]={0,1,2,0x7fffffffu,0x80000000u,0xfffffffeu,0xffffffffu,343597383u,343597384u,3951369911u,2147483646u};for(int n:{0,1,17,129,513}){std::vector<unsigned>input(n?n:1);std::vector<float>output((n?n:1)+16,-12345);for(unsigned i=0;i<input.size();i++)input[i]=i<11?special[i]:i*2654435761u;unsigned*in;float*out;CHECK(cudaMalloc(&in,input.size()*4));CHECK(cudaMalloc(&out,output.size()*4));CHECK(cudaMemcpy(in,input.data(),input.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(out,output.data(),output.size()*4,cudaMemcpyHostToDevice));inverseCNDKernel<<<2,128>>>(out,in,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),out,output.size()*4,cudaMemcpyDeviceToHost));bool pass=true;double maximum=0;for(unsigned i=0;i<output.size();i++){double expected=i<unsigned(n)?reference(input[i]):-12345,error=fabs(output[i]-expected);maximum=fmax(maximum,error);if(!std::isfinite(output[i])||error>.00002)pass=false;}printf("n=%d result + guards: %s maxError=%.9g\n",n,pass?"PASS":"FAIL",maximum);failures+=!pass;CHECK(cudaFree(in));CHECK(cudaFree(out));}return failures?1:0;}
