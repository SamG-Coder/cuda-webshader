#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "native-tile-phases.cu"
int main(){
 int *device=nullptr,values[128];
 if(cudaMalloc(&device,sizeof(values))!=cudaSuccess)return 1;
 tile_phases<<<1,128>>>(device);
 if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
 if(cudaMemcpy(values,device,sizeof(values),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
 cudaFree(device);int errors=0;
 for(int i=0;i<128;i++){int index=i%16,row=index/4;if(values[i]!=index+1+2*row*(row+1))errors++;}
 std::printf("{\"mode\":\"native CUDA\",\"values\":128,\"errors\":%d}\n",errors);
 return errors?4:0;
}
