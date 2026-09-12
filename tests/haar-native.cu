// Original project reference harness, MIT. NVIDIA kernel retains its BSD notice.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include "haar-kernel.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(int n:{4,32,1024,4096}){
 std::vector<float> initial(n),reference(n),actual(n+16,-12345);for(int i=0;i<n;i++)initial[i]=(i%29-14)*0.125f;auto current=initial;
 for(int length=n;length>1;length/=2){std::vector<float> next(length/2);for(int i=0;i<length/2;i++){next[i]=(double(current[2*i])+current[2*i+1])*std::sqrt(0.5);reference[length/2+i]=(double(current[2*i])-current[2*i+1])*std::sqrt(0.5);}current=next;}reference[0]=current[0];
 float *input,*output,*approx;CHECK(cudaMalloc(&input,(n+16)*4));CHECK(cudaMalloc(&output,(n+16)*4));CHECK(cudaMalloc(&approx,(n+16)*4));CHECK(cudaMemcpy(input,actual.data(),(n+16)*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(input,initial.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(output,actual.data(),(n+16)*4,cudaMemcpyHostToDevice));
 for(int length=n;length>1;){int block=length>1024?512:length/2,groups=length/(2*block),levels=0;for(int v=2*block;v>1;v/=2)levels++;dwtHaar1D<<<groups,block,(2*block+2*block/16)*4>>>(input,output,approx,levels,length/2,block);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(groups==1?output:input,approx,groups*4,cudaMemcpyDeviceToDevice));length=groups;}
 CHECK(cudaMemcpy(actual.data(),output,(n+16)*4,cudaMemcpyDeviceToHost));float error=0;for(int i=0;i<n+16;i++){float expected=i<n?reference[i]:-12345;error=fmaxf(error,fabsf(actual[i]-expected));if(!std::isfinite(actual[i])||fabsf(actual[i]-expected)>3e-5){printf("FAIL n=%d index=%d actual=%g expected=%g\n",n,i,actual[i],expected);return 1;}}printf("n=%d PASS original Haar kernel, full transform, output guards, max error %.9g\n",n,error);CHECK(cudaFree(input));CHECK(cudaFree(output));CHECK(cudaFree(approx));
 }return 0;}
