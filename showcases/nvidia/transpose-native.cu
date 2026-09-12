// Original verification harness: MIT. Included NVIDIA kernels: BSD-3-Clause.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include "kernels/18.cu"
#include "kernels/19.cu"
#include "kernels/35.cu"
#define CHECK(call) do {cudaError_t error=(call); if(error!=cudaSuccess){printf("%s\n",cudaGetErrorString(error));return 2;}}while(0)
int main() {
 const int shapes[][2]={{32,32},{64,96},{96,64},{128,128}};
 for(int variant=0;variant<3;variant++)for(const auto& shape:shapes){
  int width=shape[0],height=shape[1],n=width*height;size_t bytes=(n+16)*sizeof(float);
  std::vector<float> input(n+16),output(n+16,-12345.0f);
  for(int i=0;i<n+16;i++)input[i]=i/8.0f-width;
  float *a,*b;CHECK(cudaMalloc(&a,bytes));CHECK(cudaMalloc(&b,bytes));
  CHECK(cudaMemcpy(a,input.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,output.data(),bytes,cudaMemcpyHostToDevice));
  if(variant==0)transposeCoalesced<<<dim3(width/32,height/32),dim3(32,16)>>>(b,a,width,height);
  else if(variant==1)transposeNoBankConflicts<<<dim3(width/32,height/32),dim3(32,16)>>>(b,a,width,height);
  else transposeNaive<<<dim3(width/32,height/32),dim3(32,16)>>>(b,a,width,height);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),b,bytes,cudaMemcpyDeviceToHost));
  for(int y=0;y<height;y++)for(int x=0;x<width;x++)if(output[x*height+y]!=input[y*width+x]){printf("FAIL output\n");return 1;}
  for(int i=n;i<n+16;i++)if(output[i]!=-12345.0f){printf("FAIL guard\n");return 1;}
  CHECK(cudaFree(a));CHECK(cudaFree(b));printf("%s %dx%d PASS (all values and 16 guard values)\n",variant==2?"transposeNaive":variant?"transposeNoBankConflicts":"transposeCoalesced",width,height);
 }
}
