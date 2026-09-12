// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include "packed-launch.cu"
int main(){const unsigned words[]={0x80ff0100u,0x12345678u,0xffffffffu,0};const bool choose[]={true,false,true,false};unsigned* output;unsigned actual[8];for(auto& v:actual)v=0xdeadbeef;if(cudaMalloc(&output,sizeof(actual))!=cudaSuccess)return 2;if(cudaMemcpy(output,actual,sizeof(actual),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;for(unsigned i=0;i<4;i++){auto w=words[i];packedLaunch<<<1,1>>>((uchar4*)output,make_uchar4(w,w>>8,w>>16,w>>24),choose[i],i);}auto status=cudaMemcpy(actual,output,sizeof(actual),cudaMemcpyDeviceToHost);cudaFree(output);if(status!=cudaSuccess)return 2;const unsigned expected[]={0x80ff0100u,0x78563412u,0xffffffffu,0};for(int i=0;i<8;i++)if(actual[i]!=(i<4?expected[i]:0xdeadbeef))return 1;printf("PASS native uchar4/bool launch parameters: %08x %08x %08x %08x; guards unchanged\n",actual[0],actual[1],actual[2],actual[3]);}
