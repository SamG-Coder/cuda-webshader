// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include "fluids-advectVelocity_k.cuh"
#include "fluids-diffuseProject_k.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA error line %d: %s\n",__LINE__,cudaGetErrorString(e));return 2;}}while(0)
bool save(const char*name,const void*data,size_t bytes){FILE*f=fopen(name,"wb");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;fclose(f);return ok;}
__global__ void traceAdvection(cudaTextureObject_t texture,float4*output,float dt){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=19*13)return;int x=i%19,y=i/19;float2 v=tex2D<float2>(texture,(float)x,(float)y);float px=(x+.5f)-(dt*v.x*19),py=(y+.5f)-(dt*v.y*13);float2 sampled=tex2D<float2>(texture,px,py);output[i]=make_float4(sampled.x,sampled.y,px,py);}
int main(){
 const int width=19,height=13,pitch=24,lb=3;const int records=width*height;
 std::vector<float2> input(records);for(int i=0;i<records;i++)input[i]=make_float2(sinf(i*.13f)*.7f,cosf(i*.17f)*.6f);
 if(!save("reports/fluids-velocity-input.bin",input.data(),records*8))return 2;
 float2* v;float *x,*y;CHECK(cudaMalloc(&v,records*8));CHECK(cudaMemcpy(v,input.data(),records*8,cudaMemcpyHostToDevice));CHECK(cudaMalloc(&x,(pitch*height+16)*4));CHECK(cudaMalloc(&y,(pitch*height+16)*4));
 cudaArray* array;auto desc=cudaCreateChannelDesc<float2>();CHECK(cudaMallocArray(&array,&desc,width,height));CHECK(cudaMemcpy2DToArray(array,0,0,input.data(),width*8,width*8,height,cudaMemcpyHostToDevice));
 cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;cudaTextureDesc settings={};settings.normalizedCoords=false;settings.filterMode=cudaFilterModeLinear;settings.addressMode[0]=cudaAddressModeWrap;settings.readMode=cudaReadModeElementType;cudaTextureObject_t texture;CHECK(cudaCreateTextureObject(&texture,&resource,&settings,nullptr));
 std::vector<float> output(pitch*height+16,-77);float4* trace;CHECK(cudaMalloc(&trace,records*16));
 for(int mode=0;mode<3;mode++){
  CHECK(cudaMemcpy(x,output.data(),output.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(y,output.data(),output.size()*4,cudaMemcpyHostToDevice));
  float dt=mode==0?0:mode==1?.09f:1.f;advectVelocity_k<<<dim3(2,2),dim3(16,4)>>>(v,x,y,width,pitch,height,dt,lb,texture);CHECK(cudaGetLastError());
  traceAdvection<<<2,128>>>(texture,trace,dt);std::vector<float4> traced(records);CHECK(cudaMemcpy(traced.data(),trace,records*16,cudaMemcpyDeviceToHost));char tracePath[128];snprintf(tracePath,sizeof(tracePath),"reports/fluids-trace-%d.bin",mode);if(!save(tracePath,traced.data(),records*16))return 2;
  for(int axis=0;axis<2;axis++){std::vector<float> result(output.size());CHECK(cudaMemcpy(result.data(),axis?y:x,result.size()*4,cudaMemcpyDeviceToHost));char path[128];snprintf(path,sizeof(path),"reports/fluids-advect-%d-%d-native.bin",mode,axis);if(!save(path,result.data(),result.size()*4))return 2;}
  printf("Captured original advection: dt=%g, 19x13 field, 24-float row stride, unnormalized linear CUDA texture.\n",dt);
 }
 CHECK(cudaFree(trace));CHECK(cudaFree(x));CHECK(cudaFree(y));CHECK(cudaFree(v));CHECK(cudaDestroyTextureObject(texture));CHECK(cudaFreeArray(array));
 float2 *cx,*cy;CHECK(cudaMalloc(&cx,records*8));CHECK(cudaMalloc(&cy,records*8));CHECK(cudaMemcpy(cx,input.data(),records*8,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(cy,input.data(),records*8,cudaMemcpyHostToDevice));
 diffuseProject_k<<<dim3(2,2),dim3(16,4)>>>(cx,cy,width,height,.09f,.0025f,lb);
 for(int axis=0;axis<2;axis++){CHECK(cudaMemcpy(input.data(),axis?cy:cx,records*8,cudaMemcpyDeviceToHost));char path[128];snprintf(path,sizeof(path),"reports/fluids-project-%d-native.bin",axis);if(!save(path,input.data(),records*8))return 2;}
 CHECK(cudaFree(cx));CHECK(cudaFree(cy));puts("Captured original diffusion/projection.");return 0;
}
