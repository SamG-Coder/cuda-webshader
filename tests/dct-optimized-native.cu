// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include "dct-float-kernel.cuh"
#include "dct-optimized-kernel.cuh"
#define CHECK(x) do{if((x)!=cudaSuccess)return 2;}while(0)
int main(){for(int c=0;c<2;c++){int w=c?512:64,h=c?512:32,stride=c?512:72,n=stride*h;std::vector<float> input(n+16,-12345),result(n+16,-12345);if(c){FILE*f=fopen("reports/dct-teapot-input.bin","rb");if(!f||fread(input.data(),4,n,f)!=n)return 2;fclose(f);}else for(int y=0;y<h;y++)for(int x=0;x<w;x++)input[y*stride+x]=float((x*37+y*13)%256)-128;float*a,*b;CHECK(cudaMalloc(&a,input.size()*4));CHECK(cudaMalloc(&b,input.size()*4));CHECK(cudaMemcpy(a,input.data(),input.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,result.data(),result.size()*4,cudaMemcpyHostToDevice));for(int pass=0;pass<3;pass++){if(pass==0)CUDAkernel2DCT<<<dim3(w/32,h/16),dim3(8,4,2)>>>(b,a,stride);if(pass==1)CUDAkernelQuantizationFloat<<<dim3(w/8,h/8),dim3(8,8)>>>(b,stride);if(pass==2)CUDAkernel2IDCT<<<dim3(w/32,h/16),dim3(8,4,2)>>>(a,b,stride);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(result.data(),pass==2?a:b,result.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<n+16;i++){if(!std::isfinite(result[i]))return 3;if((i>=n||i%stride>=w)&&result[i]!=-12345)return 3;}char path[128];snprintf(path,sizeof(path),"reports/dct-optimized-native-%d-%d.bin",c,pass);FILE*f=fopen(path,"wb");if(!f||fwrite(result.data(),4,result.size(),f)!=result.size())return 2;fclose(f);printf("PASS optimized DCT case %d stage %d: %dx%d, stride %d, guards intact.\n",c,pass,w,h,stride);}CHECK(cudaFree(a));CHECK(cudaFree(b));}}
