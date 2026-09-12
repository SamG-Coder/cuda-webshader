// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <cuda/std/utility>
#include <vector>
#include <cstdio>
#include "helper_math.h"
#include "particle-integration-kernel.cuh"
#define CHECK(x) do{if((x)!=cudaSuccess){printf("CUDA failure %d\n",__LINE__);return 2;}}while(0)
bool save(const char*path,const void*data,size_t bytes){FILE*f=fopen(path,"wb");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;fclose(f);return ok;}
int main(){unsigned n=1024,cells=64*64*64;SimParams p={};p.gridSize=make_uint3(64);p.numCells=cells;p.numBodies=n;p.worldOrigin=make_float3(-1);p.particleRadius=1.f/64;p.cellSize=make_float3(1.f/32);p.colliderPos=make_float3(0,-.5f,0);p.colliderRadius=.2f;p.spring=.5f;p.damping=.02f;p.shear=.1f;p.gravity=make_float3(0,-.0003f,0);p.globalDamping=1;p.boundaryDamping=-.5f;CHECK(cudaMemcpyToSymbol(cudaParams,&p,sizeof(p)));std::vector<float4> pos(n),vel(n);for(unsigned i=0;i<n;i++)pos[i]=make_float4((i%16)*.0304f-.23f,((i/16)%8)*.0304f+.25f,(i/128)*.0304f-.12f,1);if(!save("reports/particle-simulation-input.bin",pos.data(),n*16))return 2;float4 *dp,*dv,*sp,*sv;unsigned *hash,*index,*start,*end;CHECK(cudaMalloc(&dp,n*16));CHECK(cudaMalloc(&dv,n*16));CHECK(cudaMalloc(&sp,n*16));CHECK(cudaMalloc(&sv,n*16));CHECK(cudaMalloc(&hash,n*4));CHECK(cudaMalloc(&index,n*4));CHECK(cudaMalloc(&start,cells*4));CHECK(cudaMalloc(&end,cells*4));CHECK(cudaMemcpy(dp,pos.data(),n*16,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dv,vel.data(),n*16,cudaMemcpyHostToDevice));auto a=thrust::device_ptr<float4>(dp),b=thrust::device_ptr<float4>(dv);for(unsigned frame=1;frame<=64;frame++){thrust::for_each(thrust::make_zip_iterator(a,b),thrust::make_zip_iterator(a+n,b+n),integrate_functor(.5f));calcHashD<<<8,128>>>(hash,index,dp,n);thrust::sort_by_key(thrust::device_ptr<unsigned>(hash),thrust::device_ptr<unsigned>(hash+n),thrust::device_ptr<unsigned>(index));CHECK(cudaMemset(start,255,cells*4));CHECK(cudaMemset(end,0,cells*4));reorderDataAndFindCellStartD<<<8,128,516>>>(start,end,sp,sv,hash,index,dp,dv,n);collideD<<<8,128>>>(dv,sp,sv,index,start,end,n);CHECK(cudaGetLastError());if(frame==1||frame==8||frame==32||frame==64){CHECK(cudaMemcpy(pos.data(),dp,n*16,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(vel.data(),dv,n*16,cudaMemcpyDeviceToHost));char path[128];snprintf(path,sizeof(path),"reports/particle-simulation-%u-positions.bin",frame);if(!save(path,pos.data(),n*16))return 2;snprintf(path,sizeof(path),"reports/particle-simulation-%u-velocities.bin",frame);if(!save(path,vel.data(),n*16))return 2;printf("PASS native original Thrust integration and collisions at frame %u.\n",frame);}}CHECK(cudaDeviceSynchronize());CHECK(cudaFree(dp));CHECK(cudaFree(dv));CHECK(cudaFree(sp));CHECK(cudaFree(sv));CHECK(cudaFree(hash));CHECK(cudaFree(index));CHECK(cudaFree(start));CHECK(cudaFree(end));}
