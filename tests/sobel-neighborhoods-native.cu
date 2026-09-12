// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "sobel-compute.cuh"
#include "sobel-neighborhoods.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){const unsigned n=4099,size=4120;std::vector<unsigned char> input(n*9);for(unsigned i=0;i<input.size();i++)input[i]=(i*37+(i/9)*13)&255;unsigned char *src,*dst;CHECK(cudaMalloc(&src,input.size()));CHECK(cudaMalloc(&dst,size));CHECK(cudaMemcpy(src,input.data(),input.size(),cudaMemcpyHostToDevice));float scales[]={-1,0,.25f,1,4};for(int k=0;k<5;k++){std::vector<unsigned char> output(size,165);CHECK(cudaMemcpy(dst,output.data(),size,cudaMemcpyHostToDevice));sobelNeighborhoods<<<(n+127)/128,128>>>(src,dst,n,scales[k]);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),dst,size,cudaMemcpyDeviceToHost));for(unsigned i=0;i<n;i++){auto p=input.data()+i*9;int h=p[2]+2*p[5]+p[8]-p[0]-2*p[3]-p[6],v=p[0]+2*p[1]+p[2]-p[6]-2*p[7]-p[8],value=int(scales[k]*(abs(h)+abs(v)));value=value<0?0:value>255?255:value;if(output[i]!=value)return 3;}for(unsigned i=n;i<size;i++)if(output[i]!=165)return 3;char path[128];snprintf(path,sizeof(path),"reports/sobel-neighborhoods-native-%d.bin",k);FILE* f=fopen(path,"wb");if(!f||fwrite(output.data(),1,size,f)!=size)return 2;fclose(f);printf("PASS original ComputeSobel: scale=%g, %u neighborhoods and guard bytes.\n",scales[k],n);}CHECK(cudaFree(src));CHECK(cudaFree(dst));}
