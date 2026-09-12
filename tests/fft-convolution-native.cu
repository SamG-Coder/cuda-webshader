// SPDX-License-Identifier: MIT
// Capture original NVIDIA test0 with its original seed and full-size dimensions.
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include "convolutionFFT2D.cu"
#include "convolutionFFT2D_gold.cpp"
#define FFT(x) do { if((x)!=CUFFT_SUCCESS) return 2; } while(0)
bool save(const char* path,const std::vector<float>& data){FILE*f=fopen(path,"wb");if(!f)return false;bool ok=fwrite(data.data(),sizeof(float),data.size(),f)==data.size();fclose(f);return ok;}
int main(){
 for(int test=0;test<2;test++){
  const int w=test?2000:37,h=test?2000:19,fw=test?2048:64,fh=test?2048:32,kw=6,kh=7,kx=4,ky=3,n=fw*fh,sn=fh*(fw/2+1);
  std::vector<float> input(w*h),kernel(kw*kh),output(n),cpu(w*h);
  srand(2010);for(auto&v:input)v=float(rand()%16);for(auto&v:kernel)v=float(rand()%16);
  char path[200];auto capture=[&](const char*kind,const std::vector<float>&v){snprintf(path,sizeof(path),"reports/fft-convolution-%d-%s.bin",test,kind);return save(path,v);};
  if(!capture("input",input)||!capture("kernel",kernel))return 2;
  float *di,*dk,*pd,*pk;fComplex *ds,*ks;
  checkCudaErrors(cudaMalloc(&di,input.size()*4));checkCudaErrors(cudaMalloc(&dk,kernel.size()*4));checkCudaErrors(cudaMalloc(&pd,n*4));checkCudaErrors(cudaMalloc(&pk,n*4));checkCudaErrors(cudaMalloc(&ds,sn*8));checkCudaErrors(cudaMalloc(&ks,sn*8));
  checkCudaErrors(cudaMemcpy(di,input.data(),input.size()*4,cudaMemcpyHostToDevice));checkCudaErrors(cudaMemcpy(dk,kernel.data(),kernel.size()*4,cudaMemcpyHostToDevice));checkCudaErrors(cudaMemset(pk,0,n*4));checkCudaErrors(cudaMemset(pd,0,n*4));
  padKernel(pk,dk,fh,fw,kh,kw,ky,kx);padDataClampToBorder(pd,di,fh,fw,h,w,kh,kw,ky,kx);
  cufftHandle forward,inverse;FFT(cufftPlan2d(&forward,fh,fw,CUFFT_R2C));FFT(cufftPlan2d(&inverse,fh,fw,CUFFT_C2R));
  FFT(cufftExecR2C(forward,pk,(cufftComplex*)ks));FFT(cufftExecR2C(forward,pd,(cufftComplex*)ds));modulateAndNormalize(ds,ks,fh,fw,1);FFT(cufftExecC2R(inverse,(cufftComplex*)ds,pd));
  checkCudaErrors(cudaMemcpy(output.data(),pd,n*4,cudaMemcpyDeviceToHost));if(!capture("output",output))return 2;
  convolutionClampToBorderCPU(cpu.data(),input.data(),kernel.data(),h,w,kh,kw,ky,kx);
  double e2=0,r2=0,maxError=0;for(int y=0;y<h;y++)for(int x=0;x<w;x++){double expected=cpu[y*w+x],delta=output[y*fw+x]-expected;e2+=delta*delta;r2+=2*expected*expected;maxError=fmax(maxError,fabs(delta));}
  double l2=sqrt(e2/r2);printf("Original test0 %dx%d, FFT %dx%d: native versus original CPU relative L2 %.12g, maximum error %.12g\n",w,h,fw,fh,l2,maxError);if(l2>=1e-6)return 3;
  FFT(cufftDestroy(forward));FFT(cufftDestroy(inverse));for(void*p:{(void*)di,(void*)dk,(void*)pd,(void*)pk,(void*)ds,(void*)ks})checkCudaErrors(cudaFree(p));
 }
}
