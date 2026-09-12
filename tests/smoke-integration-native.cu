// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda/std/utility>
#include <vector>
#include <cstdio>
#include <utility>
#include <helper_math.h>
#include "smoke-integration-kernel.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %d: %s\n",__LINE__,cudaGetErrorString(e));return 2;}}while(0)
bool save(const char* path,const void* data,size_t bytes){FILE*f=fopen(path,"wb");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;fclose(f);return ok;}
int main(){
 const unsigned n=257;
 std::vector<float4> noise(64*64*64),pos(n),vel(n);std::vector<float> keys(n);
 FILE*f=fopen("reports/smoke-noise-input.bin","rb");if(!f)return 2;size_t got=fread(noise.data(),sizeof(float4),noise.size(),f);fclose(f);if(got!=noise.size())return 2;
 cudaArray* array;auto channel=cudaCreateChannelDesc<float4>();CHECK(cudaMalloc3DArray(&array,&channel,make_cudaExtent(64,64,64)));
 cudaMemcpy3DParms upload={};upload.srcPtr=make_cudaPitchedPtr(noise.data(),64*16,64,64);upload.dstArray=array;upload.extent=make_cudaExtent(64,64,64);upload.kind=cudaMemcpyHostToDevice;CHECK(cudaMemcpy3D(&upload));
 cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
 cudaTextureDesc settings={};settings.normalizedCoords=1;settings.filterMode=cudaFilterModeLinear;settings.addressMode[0]=settings.addressMode[1]=settings.addressMode[2]=cudaAddressModeWrap;settings.readMode=cudaReadModeElementType;
 cudaTextureObject_t texture;CHECK(cudaCreateTextureObject(&texture,&resource,&settings,nullptr));
 for(unsigned i=0;i<n;i++){pos[i]=make_float4(int(i%17)*.137f-.9f,int(i%13)*.219f-.8f,int(i%11)*.323f-.7f,(i%7)*.5f);vel[i]=make_float4((int(i%3)-1)*.003f,.002f,(int(i%5)-2)*.001f,1.f+(i%9)*.25f);}
 if(!save("reports/smoke-integration-input-positions.bin",pos.data(),n*16)||!save("reports/smoke-integration-input-velocities.bin",vel.data(),n*16))return 2;
 float4 *a,*b,*c,*d;float* dk;CHECK(cudaMalloc(&a,n*16));CHECK(cudaMalloc(&b,n*16));CHECK(cudaMalloc(&c,n*16));CHECK(cudaMalloc(&d,n*16));CHECK(cudaMalloc(&dk,n*4));
 CHECK(cudaMemcpy(a,pos.data(),n*16,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,vel.data(),n*16,cudaMemcpyHostToDevice));
 SimParams params={};params.gravity=make_float3(0,.0001f,0);params.globalDamping=.99f;params.noiseFreq=.1f;params.noiseAmp=.001f;params.noiseSpeed=make_float3(.01f,.02f,-.01f);
 for(unsigned frame=1;frame<=64;frame++){
  params.time=(frame-1)*.5f;CHECK(cudaMemcpyToSymbol(cudaParams,&params,sizeof(params)));
  auto pa=thrust::device_ptr<float4>(a),pb=thrust::device_ptr<float4>(b),pc=thrust::device_ptr<float4>(c),pd=thrust::device_ptr<float4>(d);
  thrust::for_each(thrust::make_zip_iterator(pc,pd,pa,pb),thrust::make_zip_iterator(pc+n,pd+n,pa+n,pb+n),integrate_functor(.5f,texture));
  std::swap(a,c);std::swap(b,d);
  if(frame==1||frame==8||frame==32||frame==64){
   auto pp=thrust::device_ptr<float4>(a);auto pk=thrust::device_ptr<float>(dk);
   float3 direction=frame<32?make_float3(.25f,-.5f,1):make_float3(-1,.125f,.5f);
   thrust::for_each(thrust::make_zip_iterator(pp,pk),thrust::make_zip_iterator(pp+n,pk+n),calcDepth_functor(direction));
   CHECK(cudaMemcpy(pos.data(),a,n*16,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(vel.data(),b,n*16,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(keys.data(),dk,n*4,cudaMemcpyDeviceToHost));
   char path[160];snprintf(path,sizeof(path),"reports/smoke-integration-%u-positions.bin",frame);if(!save(path,pos.data(),n*16))return 2;
   snprintf(path,sizeof(path),"reports/smoke-integration-%u-velocities.bin",frame);if(!save(path,vel.data(),n*16))return 2;
   snprintf(path,sizeof(path),"reports/smoke-integration-%u-depth.bin",frame);if(!save(path,keys.data(),n*4))return 2;
   printf("Captured original smoke Thrust integration and depth keys: frame=%u, particles=%u.\n",frame,n);
  }
 }
 CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(c));CHECK(cudaFree(d));CHECK(cudaFree(dk));CHECK(cudaDestroyTextureObject(texture));CHECK(cudaFreeArray(array));return 0;
}
