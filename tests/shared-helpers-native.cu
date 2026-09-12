// Original project regression harness, MIT.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <cstdio>
#include "shared-helpers.cu"
#include "shared-helpers-dynamic.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(bool dynamic:{false,true})for(int threads:{32,128}){int n=threads*3;std::vector<float>out(n+16,-12345);float*o;CHECK(cudaMalloc(&o,out.size()*4));CHECK(cudaMemcpy(o,out.data(),out.size()*4,cudaMemcpyHostToDevice));if(dynamic)testDynamicHelper<<<3,threads,threads*4>>>(o);else testSharedHelpers<<<3,threads>>>(o);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(out.data(),o,out.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<n;i++){int reversed=(i/threads)*threads+threads-1-i%threads;float expected=dynamic?float(reversed):reversed*1.125f+threads+3;if(out[i]!=expected)return 1;}for(int i=n;i<n+16;i++)if(out[i]!=-12345)return 1;CHECK(cudaFree(o));printf("dynamic=%d threads=%d groups=3 PASS helper tile exchange and guards\n",int(dynamic),threads);}return 0;}
