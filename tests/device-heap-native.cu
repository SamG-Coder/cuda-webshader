// MIT native reference for the same device-allocation fixture.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "device-heap.cu"
int main(){
 Allocation* records;int* status;float2* output;
 if(cudaMalloc(&records,256*sizeof(Allocation))||cudaMalloc(&status,256*sizeof(int))||cudaMalloc(&output,256*32*sizeof(float2)))return 1;
 cudaMemset(records,0,256*sizeof(Allocation));cudaMemset(output,0,256*32*sizeof(float2));
 allocate_values<<<4,64>>>(records,status);if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
 std::vector<int> codes(256);cudaMemcpy(codes.data(),status,256*sizeof(int),cudaMemcpyDeviceToHost);for(auto c:codes)if(c)return 3;
 write_values<<<4,64>>>(records);if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 read_values<<<4,64>>>(records,output);if(cudaDeviceSynchronize()!=cudaSuccess)return 5;
 std::vector<float2> values(256*32);cudaMemcpy(values.data(),output,values.size()*sizeof(float2),cudaMemcpyDeviceToHost);
 for(int i=0;i<256;i++)for(int j=0;j<32;j++){auto v=values[i*32+j];if(v.x!=(j<=i%32?float(i):0.f)||v.y!=(j<=i%32?float(j):0.f))return 6;}
 free_values<<<4,64>>>(records,status);if(cudaDeviceSynchronize()!=cudaSuccess)return 7;
 cudaMemcpy(codes.data(),status,256*sizeof(int),cudaMemcpyDeviceToHost);for(auto c:codes)if(c)return 8;
 FILE*f=fopen("reports/device-heap-native.bin","wb");if(!f)return 9;fwrite(values.data(),sizeof(float2),values.size(),f);fclose(f);
 cudaFree(records);cudaFree(status);cudaFree(output);
 printf("{\"passed\":true,\"allocations\":256,\"liveElements\":4224,\"floatComponents\":16384,\"separateLaunches\":true}\n");
}
