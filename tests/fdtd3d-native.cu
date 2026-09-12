// Original project reference harness, MIT. Kernel retains NVIDIA's BSD notice.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <array>
#include <vector>
#include <cstdio>
#include <cmath>
#include "fdtd3d-kernel.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){float coefficients[5]={0.5f,0.0625f,0.03125f,0.015625f,0.0078125f};CHECK(cudaMemcpyToSymbol(stencil,coefficients,sizeof(coefficients)));
 for(auto dims:{std::array<int,3>{32,4,3},{35,7,5},{64,8,9},{32,16,16}}){int x=dims[0],y=dims[1],z=dims[2],sy=x+8,sz=sy*(y+8),n=sz*(z+8);std::vector<float>expected(n),actual(n);for(int i=0;i<n;i++)expected[i]=(i%17-8)*0.125f;
  float *input,*output;size_t bytes=n*4;CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,bytes));CHECK(cudaMemcpy(input,expected.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(output,expected.data(),bytes,cudaMemcpyHostToDevice));
  for(int step=0;step<3;step++){auto next=expected;for(int iz=4;iz<z+4;iz++)for(int iy=4;iy<y+4;iy++)for(int ix=4;ix<x+4;ix++){int p=iz*sz+iy*sy+ix;double value=expected[p]*coefficients[0];for(int r=1;r<=4;r++)value+=coefficients[r]*(double(expected[p-r])+expected[p+r]+expected[p-r*sy]+expected[p+r*sy]+expected[p-r*sz]+expected[p+r*sz]);next[p]=float(value);}
   FiniteDifferencesKernel<<<dim3((x+31)/32,(y+3)/4),dim3(32,4)>>>(output,input,x,y,z);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(actual.data(),output,bytes,cudaMemcpyDeviceToHost));for(int i=0;i<n;i++)if(!std::isfinite(actual[i])||std::abs(actual[i]-next[i])>3e-5f){printf("FAIL %dx%dx%d step=%d index=%d\n",x,y,z,step,i);return 1;}expected=next;auto tmp=input;input=output;output=tmp;
  }CHECK(cudaFree(input));CHECK(cudaFree(output));printf("%dx%dx%d steps=3 PASS original FDTD3d, independent reference and unchanged halos\n",x,y,z);
 }return 0;
}
