// Original project reference harness, MIT. Included kernels retain NVIDIA BSD notices.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <cstdio>
#include <cmath>
namespace cg=cooperative_groups;
#include "nbody-vector-traits.cuh"
#include "nbody-rsqrt.cuh"
#include "nbody-interaction.cuh"
#include "nbody-shared-memory.cuh"
#include "nbody-integrate.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){float softening=0.125f,dt=0.002f,damping=0.999f;CHECK(cudaMemcpyToSymbol(softeningSquared,&softening,4));
 for(int threads:{32,128})for(int blocks:{1,3,4}){
  int n=threads*blocks;std::vector<float>pos((n+4)*4,-12345),vel=pos,out=pos,actual=pos,actualVel=pos;
  for(int i=0;i<n;i++){pos[i*4]=(i%11-5)*0.125f;pos[i*4+1]=(i%7-3)*0.25f;pos[i*4+2]=(i%13-6)*0.125f;pos[i*4+3]=0.5f+(i%5)*0.125f;vel[i*4]=(i%3-1)*0.01f;vel[i*4+1]=(i%5-2)*0.01f;vel[i*4+2]=0;vel[i*4+3]=1;}
  float4 *old,*next,*v;size_t bytes=pos.size()*4;CHECK(cudaMalloc(&old,bytes));CHECK(cudaMalloc(&next,bytes));CHECK(cudaMalloc(&v,bytes));CHECK(cudaMemcpy(old,pos.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(next,out.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(v,vel.data(),bytes,cudaMemcpyHostToDevice));
  for(int step=0;step<3;step++){
   out=pos;auto nextVel=vel;
   for(int i=0;i<n;i++){double acc[3]={0,0,0};for(int j=0;j<n;j++){double r[3],d=softening;for(int c=0;c<3;c++){r[c]=double(pos[j*4+c])-pos[i*4+c];d+=r[c]*r[c];}double scale=pos[j*4+3]/std::pow(d,1.5);for(int c=0;c<3;c++)acc[c]+=r[c]*scale;}
    for(int c=0;c<3;c++){nextVel[i*4+c]=float((vel[i*4+c]+acc[c]*dt)*damping);out[i*4+c]=float(double(pos[i*4+c])+double(nextVel[i*4+c])*dt);}}
   integrateBodies<float><<<blocks,threads,threads*16>>>(next,old,v,0,n,dt,damping,blocks);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(actual.data(),next,bytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(actualVel.data(),v,bytes,cudaMemcpyDeviceToHost));
   for(size_t i=0;i<pos.size();i++)if(!std::isfinite(actual[i])||!std::isfinite(actualVel[i])||std::abs(actual[i]-out[i])>3e-5f||std::abs(actualVel[i]-nextVel[i])>3e-5f){printf("FAIL n=%d step=%d index=%zu\n",n,step,i);return 1;}
   pos=out;vel=nextVel;auto tmp=old;old=next;next=tmp;
  }
  CHECK(cudaFree(old));CHECK(cudaFree(next));CHECK(cudaFree(v));printf("n=%d threads=%d steps=3 PASS original integration, independent reference and guards\n",n,threads);
 }return 0;
}
