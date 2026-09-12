// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cstdio>
#include "fluids-solver-kernels.cuh"
#define CUDA(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA error %d: %s\n",__LINE__,cudaGetErrorString(e));return 2;}}while(0)
#define FFT(x) do{if((x)!=CUFFT_SUCCESS){fprintf(stderr,"FFT error %d\n",__LINE__);return 2;}}while(0)
bool save(const char*path,const void*data,size_t bytes){FILE*f=fopen(path,"wb");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;fclose(f);return ok;}
int main(){
 for(int width:{64,512}){
  int height=width,velocityStride=width+(width==64?3:0),realStride=2*(width/2+1),packed=width/2+1,count=width*height;
  std::vector<float2> velocity(velocityStride*height,make_float2(-77,-77)),particles(count);
  for(int y=0;y<height;y++)for(int x=0;x<width;x++){velocity[y*velocityStride+x]=make_float2(0,0);particles[y*width+x]=make_float2((x+.5f)/width,(y+.5f)/height);}
  char path[160];snprintf(path,sizeof(path),"reports/fluids-solver-%d-input-velocity.bin",width);if(!save(path,velocity.data(),velocity.size()*8))return 2;
  snprintf(path,sizeof(path),"reports/fluids-solver-%d-input-particles.bin",width);if(!save(path,particles.data(),particles.size()*8))return 2;
  float2 *v,*p;float *vx,*vy;CUDA(cudaMalloc(&v,velocity.size()*8));CUDA(cudaMalloc(&p,count*8));CUDA(cudaMalloc(&vx,realStride*height*4));CUDA(cudaMalloc(&vy,realStride*height*4));CUDA(cudaMemcpy(v,velocity.data(),velocity.size()*8,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(p,particles.data(),count*8,cudaMemcpyHostToDevice));CUDA(cudaMemset(vx,0,realStride*height*4));CUDA(cudaMemset(vy,0,realStride*height*4));
  cudaArray*array;auto channel=cudaCreateChannelDesc<float2>();CUDA(cudaMallocArray(&array,&channel,width,height));cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;cudaTextureDesc settings={};settings.normalizedCoords=false;settings.filterMode=cudaFilterModeLinear;settings.addressMode[0]=cudaAddressModeWrap;settings.readMode=cudaReadModeElementType;cudaTextureObject_t texture;CUDA(cudaCreateTextureObject(&texture,&resource,&settings,nullptr));
  cufftHandle forward,inverse;FFT(cufftPlan2d(&forward,height,width,CUFFT_R2C));FFT(cufftPlan2d(&inverse,height,width,CUFFT_C2R));
  dim3 block(64,4),grid((width+63)/64,(height+63)/64),frequencyGrid((packed+63)/64,(height+63)/64);
  for(int step=1;step<=64;step++){
   addForces_k<<<1,dim3(9,9)>>>(v,width,height,width/2-4,height/2-4,.06f,.04f,4,velocityStride*8);
   CUDA(cudaMemcpy2DToArray(array,0,0,v,velocityStride*8,width*8,height,cudaMemcpyDeviceToDevice));
   advectVelocity_k<<<grid,block>>>(v,vx,vy,width,realStride,height,.09f,16,texture);
   FFT(cufftExecR2C(forward,vx,(cufftComplex*)vx));FFT(cufftExecR2C(forward,vy,(cufftComplex*)vy));
   diffuseProject_k<<<frequencyGrid,block>>>((float2*)vx,(float2*)vy,packed,height,.09f,.0025f,16);
   FFT(cufftExecC2R(inverse,(cufftComplex*)vx,vx));FFT(cufftExecC2R(inverse,(cufftComplex*)vy,vy));
   updateVelocity_k<<<grid,block>>>(v,vx,vy,width,realStride,height,16,velocityStride*8);
   advectParticles_k<<<grid,block>>>(p,v,width,height,.09f,16,velocityStride*8);
   if(step==1||step==8||step==32||step==64){
    CUDA(cudaMemcpy(velocity.data(),v,velocity.size()*8,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(particles.data(),p,count*8,cudaMemcpyDeviceToHost));
    snprintf(path,sizeof(path),"reports/fluids-solver-%d-%d-velocity.bin",width,step);if(!save(path,velocity.data(),velocity.size()*8))return 2;
    snprintf(path,sizeof(path),"reports/fluids-solver-%d-%d-particles.bin",width,step);if(!save(path,particles.data(),particles.size()*8))return 2;
    printf("Captured full native CUDA/cuFFT solver: %dx%d, step %d.\n",width,height,step);
   }
  }
  CUDA(cudaFree(v));CUDA(cudaFree(p));CUDA(cudaFree(vx));CUDA(cudaFree(vy));CUDA(cudaDestroyTextureObject(texture));CUDA(cudaFreeArray(array));FFT(cufftDestroy(forward));FFT(cufftDestroy(inverse));
 }
}
