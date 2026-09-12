// Original project regression harness, MIT.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "vector-initializers.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(unsigned n:{1u,129u,1025u}){
 std::vector<float>out(n*12+16,-12345);float4* p;CHECK(cudaMalloc(&p,out.size()*4));CHECK(cudaMemcpy(p,out.data(),out.size()*4,cudaMemcpyHostToDevice));
 testVectorInitializers<<<(n+127)/128,128>>>(p,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(out.data(),p,out.size()*4,cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<n;i++){float expected[12]={0,0,0,0,float(i)+5,0,0,0,1,2,3,4};for(int j=0;j<12;j++)if(out[i*12+j]!=expected[j])return 1;}
 for(unsigned i=n*12;i<out.size();i++)if(out[i]!=-12345)return 1;CHECK(cudaFree(p));printf("n=%u PASS vector initializers, dependent helper types and guards\n",n);
}return 0;}
