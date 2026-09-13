// MIT native CUDA reference harness.
#include <cuda_runtime.h>
#include <fstream>
#include <cstdio>
#include "switch.cu"
int main(){int input[9]={-3,-2,-1,0,1,2,3,4,5},output[36];int *a,*b;
 if(cudaMalloc(&a,sizeof(input))!=cudaSuccess||cudaMalloc(&b,sizeof(output))!=cudaSuccess)return 1;
 if(cudaMemcpy(a,input,sizeof(input),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
 switchCases<<<1,9>>>(a,b);if(cudaDeviceSynchronize()!=cudaSuccess||cudaMemcpy(output,b,sizeof(output),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
 std::ofstream file("reports/switch-native.json");file<<"[";for(int i=0;i<36;i++)file<<(i?",":"")<<output[i];file<<"]\n";
 cudaFree(a);cudaFree(b);return file?0:4;}
