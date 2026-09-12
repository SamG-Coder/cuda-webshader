// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cstdio>
#define CUDA(x) do{if((x)!=cudaSuccess){fprintf(stderr,"CUDA error %d\n",__LINE__);return 2;}}while(0)
#define FFT(x) do{if((x)!=CUFFT_SUCCESS){fprintf(stderr,"cuFFT error %d\n",__LINE__);return 2;}}while(0)
bool save(const char*path,const void*data,size_t bytes){FILE*f=fopen(path,"wb");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;fclose(f);return ok;}
int main(){
 const int dimensions[][2]={{1,1},{2,4},{8,4},{64,32},{512,512}};
 for(const auto& dims:dimensions){
  int width=dims[0],height=dims[1],stride=2*(width/2+1),n=stride*height;std::vector<float> data(n,-77),compact(width*height);
  unsigned seed=123;for(int y=0;y<height;y++)for(int x=0;x<width;x++){seed=seed*1664525u+1013904223u;data[y*stride+x]=float(int(seed>>16)-32768)/32768.f;}
  char path[160];snprintf(path,sizeof(path),"reports/real-fft-%dx%d-input.bin",width,height);if(!save(path,data.data(),n*4))return 2;
  float*gpu;CUDA(cudaMalloc(&gpu,n*4));CUDA(cudaMemcpy(gpu,data.data(),n*4,cudaMemcpyHostToDevice));cufftHandle forward,inverse;FFT(cufftPlan2d(&forward,height,width,CUFFT_R2C));FFT(cufftPlan2d(&inverse,height,width,CUFFT_C2R));
  FFT(cufftExecR2C(forward,gpu,(cufftComplex*)gpu));CUDA(cudaMemcpy(data.data(),gpu,n*4,cudaMemcpyDeviceToHost));snprintf(path,sizeof(path),"reports/real-fft-%dx%d-spectrum.bin",width,height);if(!save(path,data.data(),n*4))return 2;
  FFT(cufftExecC2R(inverse,(cufftComplex*)gpu,gpu));CUDA(cudaMemcpy(data.data(),gpu,n*4,cudaMemcpyDeviceToHost));for(int y=0;y<height;y++)for(int x=0;x<width;x++)compact[y*width+x]=data[y*stride+x];
  snprintf(path,sizeof(path),"reports/real-fft-%dx%d-inverse.bin",width,height);if(!save(path,compact.data(),compact.size()*4))return 2;
  FFT(cufftDestroy(forward));FFT(cufftDestroy(inverse));CUDA(cudaFree(gpu));printf("Captured cuFFT in-place R2C and unnormalized C2R: %dx%d, real stride %d.\n",width,height,stride);
 }
}
