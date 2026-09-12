// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>
#include "ocean-initial-spectrum.cuh"
#include "ocean-kernel.cuh"
#define CHECK(x) do{if((x)!=0){printf("CUDA/FFT error at %d\n",__LINE__);return 2;}}while(0)
bool save(const char*path,const void*p,size_t bytes){FILE*f=fopen(path,"wb");if(!f)return false;bool ok=fwrite(p,1,bytes,f)==bytes;fclose(f);return ok;}
int main(){std::vector<float2> h0(spectrumW*spectrumH),values(meshSize*meshSize);srand(1);generate_h0(h0.data());if(!save("reports/ocean-h0.bin",h0.data(),h0.size()*8))return 2;float2*initial,*spectrum,*slopes;float*heights;CHECK(cudaMalloc(&initial,h0.size()*8));CHECK(cudaMalloc(&spectrum,values.size()*8));CHECK(cudaMalloc(&slopes,values.size()*8));CHECK(cudaMalloc(&heights,values.size()*4));CHECK(cudaMemcpy(initial,h0.data(),h0.size()*8,cudaMemcpyHostToDevice));cufftHandle plan;CHECK(cufftPlan2d(&plan,meshSize,meshSize,CUFFT_C2C));float times[]={0.f,1.25f,7.5f};for(int c=0;c<3;c++){generateSpectrumKernel<<<dim3(32,32),dim3(8,8)>>>(initial,spectrum,spectrumW,meshSize,meshSize,times[c],patchSize);CHECK(cudaGetLastError());CHECK(cudaMemcpy(values.data(),spectrum,values.size()*8,cudaMemcpyDeviceToHost));char path[128];snprintf(path,sizeof(path),"reports/ocean-native-%d-spectrum.bin",c);if(!save(path,values.data(),values.size()*8))return 2;CHECK(cufftExecC2C(plan,spectrum,spectrum,CUFFT_INVERSE));CHECK(cudaMemcpy(values.data(),spectrum,values.size()*8,cudaMemcpyDeviceToHost));snprintf(path,sizeof(path),"reports/ocean-native-%d-spatial.bin",c);if(!save(path,values.data(),values.size()*8))return 2;updateHeightmapKernel<<<dim3(32,32),dim3(8,8)>>>(heights,spectrum,meshSize);calculateSlopeKernel<<<dim3(32,32),dim3(8,8)>>>(heights,slopes,meshSize,meshSize);CHECK(cudaGetLastError());CHECK(cudaMemcpy(values.data(),heights,values.size()*4,cudaMemcpyDeviceToHost));snprintf(path,sizeof(path),"reports/ocean-native-%d-height.bin",c);if(!save(path,values.data(),values.size()*4))return 2;CHECK(cudaMemcpy(values.data(),slopes,values.size()*8,cudaMemcpyDeviceToHost));snprintf(path,sizeof(path),"reports/ocean-native-%d-slope.bin",c);if(!save(path,values.data(),values.size()*8))return 2;printf("PASS native ocean at t=%g: original spectrum, cuFFT inverse, real heights and slopes.\n",times[c]);}CHECK(cufftDestroy(plan));CHECK(cudaFree(initial));CHECK(cudaFree(spectrum));CHECK(cudaFree(slopes));CHECK(cudaFree(heights));}
