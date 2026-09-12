// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
typedef unsigned int uint;
#include "cuda-extent.cu"
int main(){int input[]={-2147483647-1,-1,0,1,7,8,2147483647};int* dInput;uint* dOutput;cudaMalloc(&dInput,sizeof(input));cudaMalloc(&dOutput,56*4);cudaMemcpy(dInput,input,sizeof(input),cudaMemcpyHostToDevice);uint widths[]={0,8,4294967295u};std::vector<uint> all;for(uint w:widths){cudaExtent size=make_cudaExtent(w,3,4);extentProbe<<<1,32>>>(dInput,dOutput,size,7);std::vector<uint> out(56);if(cudaMemcpy(out.data(),dOutput,56*4,cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;for(int i=0;i<7;i++){size_t x=input[i];uint expected[]={x<size.width,x<=size.width,x>size.width,x>=size.width,x==size.width,x!=size.width,size.height<size.depth,size.width>=(uint)input[i]};for(int c=0;c<8;c++)if(out[i*8+c]!=expected[c])return 1;}all.insert(all.end(),out.begin(),out.end());}cudaFree(dInput);cudaFree(dOutput);FILE* f=fopen("reports/cuda-extent-native.bin","wb");if(!f)return 2;fwrite(all.data(),4,all.size(),f);fclose(f);printf("PASS %zu extent comparison results, negative signed coordinates and full u32 component range\n",all.size());}
