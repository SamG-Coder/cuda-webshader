// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "surface3d.cu"
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){cudaArray_t array;auto desc=cudaCreateChannelDesc<float>();auto extent=make_cudaExtent(8,8,4);CHECK(cudaMalloc3DArray(&array,&desc,extent,cudaArraySurfaceLoadStore));cudaResourceDesc r={};r.resType=cudaResourceTypeArray;r.res.array.array=array;cudaSurfaceObject_t surface;CHECK(cudaCreateSurfaceObject(&surface,&r));writeVolume<<<dim3(2,2,2),dim3(4,4,2)>>>(surface);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());std::vector<float> out(256);cudaMemcpy3DParms copy={};copy.srcArray=array;copy.dstPtr=make_cudaPitchedPtr(out.data(),8*4,8,8);copy.extent=extent;copy.kind=cudaMemcpyDeviceToHost;CHECK(cudaMemcpy3D(&copy));for(int i=0;i<256;i++)if(out[i]!=i*.125f-16.f){printf("FAIL voxel %d\n",i);return 1;}CHECK(cudaDestroySurfaceObject(surface));CHECK(cudaFreeArray(array));printf("PASS 256 float surface voxels; global XYZ writes across eight workgroups\n");}
