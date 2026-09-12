// SPDX-License-Identifier: MIT
// Capture original NVIDIA custom test1/test2 with original seed and dimensions.
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
 for(int variant=1;variant<=2;variant++)for(int test=0;test<2;test++){
  const int w=test?2000:37,h=test?2000:19,fw=test?2048:64,fh=test?2048:32,kw=6,kh=7,kx=4,ky=3,n=fw*fh,sn=fh*(fw/2+16);
  std::vector<float> input(w*h),kernel(kw*kh),output(n),cpu(w*h);
  srand(2010);for(auto&v:input)v=float(rand()%16);for(auto&v:kernel)v=float(rand()%16);
  char path[200];auto capture=[&](const char*kind,const std::vector<float>&v){snprintf(path,sizeof(path),"reports/fft-custom-full-%d-%d-%s.bin",variant,test,kind);return save(path,v);};

  float *di,*dk,*pd,*pk;fComplex *ds,*ks,*ds0,*ks0;
  checkCudaErrors(cudaMalloc(&di,input.size()*4));checkCudaErrors(cudaMalloc(&dk,kernel.size()*4));checkCudaErrors(cudaMalloc(&pd,n*4));checkCudaErrors(cudaMalloc(&pk,n*4));checkCudaErrors(cudaMalloc(&ds,sn*8));checkCudaErrors(cudaMalloc(&ks,sn*8));
  checkCudaErrors(cudaMemcpy(di,input.data(),input.size()*4,cudaMemcpyHostToDevice));checkCudaErrors(cudaMemcpy(dk,kernel.data(),kernel.size()*4,cudaMemcpyHostToDevice));checkCudaErrors(cudaMemset(pk,0,n*4));checkCudaErrors(cudaMemset(pd,0,n*4));
  checkCudaErrors(cudaMalloc(&ds0,n*4));checkCudaErrors(cudaMalloc(&ks0,n*4));
  padKernel(pk,dk,fh,fw,kh,kw,ky,kx);padDataClampToBorder(pd,di,fh,fw,h,w,kh,kw,ky,kx);
  cufftHandle plan;FFT(cufftPlan2d(&plan,fh,fw/2,CUFFT_C2C));
  FFT(cufftExecC2C(plan,(cufftComplex*)pk,(cufftComplex*)ks0,CUFFT_FORWARD));FFT(cufftExecC2C(plan,(cufftComplex*)pd,(cufftComplex*)ds0,CUFFT_FORWARD));
  if(variant==1){checkCudaErrors(cudaMemset(ds,0,sn*8));checkCudaErrors(cudaMemset(ks,0,sn*8));spPostprocess2D(ks,ks0,fh,fw/2,16,-1);spPostprocess2D(ds,ds0,fh,fw/2,16,-1);modulateAndNormalize(ds,ks,fh,fw,16);spPreprocess2D(ds0,ds,fh,fw/2,16,1);}
  else spProcess2D(ds0,ds0,ks0,fh,fw/2,-1);
  FFT(cufftExecC2C(plan,(cufftComplex*)ds0,(cufftComplex*)pd,CUFFT_INVERSE));
  checkCudaErrors(cudaMemcpy(output.data(),pd,n*4,cudaMemcpyDeviceToHost));if(!capture("output",output))return 2;
  convolutionClampToBorderCPU(cpu.data(),input.data(),kernel.data(),h,w,kh,kw,ky,kx);
  double e2=0,r2=0,maxError=0;for(int y=0;y<h;y++)for(int x=0;x<w;x++){double expected=cpu[y*w+x],delta=output[y*fw+x]-expected;e2+=delta*delta;r2+=2*expected*expected;maxError=fmax(maxError,fabs(delta));}
  double l2=sqrt(e2/r2);printf("Original custom test%d %dx%d, FFT %dx%d: native versus original CPU relative L2 %.12g, maximum error %.12g\n",variant,w,h,fw,fh,l2,maxError);if(l2>=1e-6)return 3;
  FFT(cufftDestroy(plan));for(void*p:{(void*)di,(void*)dk,(void*)pd,(void*)pk,(void*)ds,(void*)ks,(void*)ds0,(void*)ks0})checkCudaErrors(cudaFree(p));
 }
}
