// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <vector>
#include "dxt-colors-kernels.cuh"
bool save(const char*p,const void*d,size_t bytes){FILE*f=fopen(p,"wb");if(!f)return false;bool ok=fwrite(d,1,bytes,f)==bytes;fclose(f);return ok;}
int main(){unsigned char*image=nullptr;unsigned width,height;if(!sdkLoadPPM4ub(".local/nvidia-audit/cpp/5_Domain_Specific/dxtc/data/teapot512_std.ppm",&image,&width,&height))return 2;const unsigned n=width*height,blocks=n/16;std::vector<uint> input(n);for(unsigned by=0;by<height/4;by++)for(unsigned bx=0;bx<width/4;bx++)for(unsigned i=0;i<16;i++)input[(by*width/4+bx)*16+i]=((uint*)image)[(by*4+i/4)*width+bx*4+i%4];free(image);if(!save("reports/dxt-colors-input.bin",input.data(),n*4))return 2;uint*di;float4*dc,*ds;int*dr;checkCudaErrors(cudaMalloc(&di,n*4));checkCudaErrors(cudaMalloc(&dc,n*16));checkCudaErrors(cudaMalloc(&ds,n*16));checkCudaErrors(cudaMalloc(&dr,n*4));checkCudaErrors(cudaMemcpy(di,input.data(),n*4,cudaMemcpyHostToDevice));dxtColors<<<37,64>>>(di,dc,ds,dr,0);dxtColors<<<blocks-37,64>>>(di,dc,ds,dr,37);std::vector<float4> colors(n),sums(n);std::vector<int> ranks(n);checkCudaErrors(cudaMemcpy(colors.data(),dc,n*16,cudaMemcpyDeviceToHost));checkCudaErrors(cudaMemcpy(sums.data(),ds,n*16,cudaMemcpyDeviceToHost));checkCudaErrors(cudaMemcpy(ranks.data(),dr,n*4,cudaMemcpyDeviceToHost));if(!save("reports/dxt-colors-native.bin",colors.data(),n*16)||!save("reports/dxt-sums-native.bin",sums.data(),n*16)||!save("reports/dxt-ranks-native.bin",ranks.data(),n*4))return 2;for(void*p:{(void*)di,(void*)dc,(void*)ds,(void*)dr})checkCudaErrors(cudaFree(p));printf("Captured original colour loading/sums/covariance/ranking at 64 threads for all %u blocks, with split offset launch.\n",blocks);}
