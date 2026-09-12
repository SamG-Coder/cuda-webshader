// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "byte-stores.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int widths[]={1,3,31,61},heights[]={1,5,9,65};for(int k=0;k<4;k++){int w=widths[k],h=heights[k],pitch=w+3,n=w*h,size=((h*pitch+10)/4)*4;std::vector<unsigned char> input(n),output(size,165),expected=output;for(int i=0;i<n;i++){input[i]=(i*37+251)&255;expected[(i/w)*pitch+i%w+1]=((input[i]+18)&255)^165;}unsigned char *src,*dst;CHECK(cudaMalloc(&src,n));CHECK(cudaMalloc(&dst,size));CHECK(cudaMemcpy(src,input.data(),n,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dst,output.data(),size,cudaMemcpyHostToDevice));bytePixels<<<(n+127)/128,128>>>(src,dst,w,h,pitch);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),dst,size,cudaMemcpyDeviceToHost));if(output!=expected)return 3;char path[128];snprintf(path,sizeof(path),"reports/byte-stores-native-%d.bin",k);FILE* f=fopen(path,"wb");if(!f||fwrite(output.data(),1,size,f)!=size)return 2;fclose(f);printf("PASS %dx%d pitch=%d: stores, helper compound updates, rebasing, wraparound and guard bytes.\n",w,h,pitch);CHECK(cudaFree(src));CHECK(cudaFree(dst));}return 0;}
