// MIT native reference for compiler prerequisites.
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include "chrono-rhs-primitives.cu"
int main(){float input[4]={1.e8f,1.f,-1.e8f,-2.75f},output[128];float *a,*b;if(cudaMalloc(&a,sizeof(input))!=cudaSuccess||cudaMalloc(&b,sizeof(output))!=cudaSuccess)return 1;cudaMemcpy(a,input,sizeof(input),cudaMemcpyHostToDevice);prerequisites<<<1,32>>>(a,b);if(cudaDeviceSynchronize()!=cudaSuccess||cudaMemcpy(output,b,sizeof(output),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;std::ofstream f("reports/chrono-rhs-primitives-native.bin",std::ios::binary);f.write((char*)output,sizeof(output));cudaFree(a);cudaFree(b);return f?0:3;}
