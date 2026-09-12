#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "native-tiles.cu"
int main(){int *device=nullptr,values[640];if(cudaMalloc(&device,sizeof(values))!=cudaSuccess)return 1;native_tiles<<<1,128>>>(device);if(cudaDeviceSynchronize()!=cudaSuccess)return 2;if(cudaMemcpy(values,device,sizeof(values),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;cudaFree(device);int errors=0;for(int i=0;i<128;i++){int warp=i/32,lane=i%32,expected[5]={warp+1,warp*100+7,lane+1,(17+warp*25+31)/32,lane};for(int j=0;j<5;j++)if(values[i*5+j]!=expected[j])errors++;}std::printf("{\"mode\":\"native CUDA\",\"tiles\":4,\"values\":640,\"errors\":%d}\n",errors);return errors?4:0;}
