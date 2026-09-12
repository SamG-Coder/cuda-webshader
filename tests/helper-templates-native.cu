// Compiler feature regression harness, MIT.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
using uint=unsigned int;
#include "helper-templates.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(uint n:{1u,129u,1025u}){std::vector<float>out(n*4+16,-12345);std::vector<uint>bits(n+16,0xdeadbeef);float*o;uint*b;CHECK(cudaMalloc(&o,out.size()*4));CHECK(cudaMalloc(&b,bits.size()*4));CHECK(cudaMemcpy(o,out.data(),out.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,bits.data(),bits.size()*4,cudaMemcpyHostToDevice));helperTemplates<<<(n+127)/128,128>>>(o,b,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(out.data(),o,out.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bits.data(),b,bits.size()*4,cudaMemcpyDeviceToHost));
for(uint i=0;i<n;i++){float x=(int(i%13)-6)*0.25f,expected[]={x*x+x,x*3,2,x+1};for(int j=0;j<4;j++)if(out[i*4+j]!=expected[j])return 1;uint v=0x80000000u+i;if(bits[i]!=v*v+v)return 1;}
for(uint i=n*4;i<out.size();i++)if(out[i]!=-12345)return 1;for(uint i=n;i<bits.size();i++)if(bits[i]!=0xdeadbeef)return 1;CHECK(cudaFree(o));CHECK(cudaFree(b));printf("n=%u PASS template output, unsigned wraparound and guards\n",n);}return 0;}
