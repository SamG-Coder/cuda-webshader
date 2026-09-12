#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "../.local/raytracing-cuda/material.h"
#include "pathtracer-material-kernels.cuh"
int main(){float*out;unsigned int*bits;checkCudaErrors(cudaMalloc(&out,512*10*4));checkCudaErrors(cudaMalloc(&bits,512*4));check_material<<<8,64>>>(out,bits);checkCudaErrors(cudaDeviceSynchronize());std::vector<float>v(512*10);std::vector<unsigned int>b(512);checkCudaErrors(cudaMemcpy(v.data(),out,v.size()*4,cudaMemcpyDeviceToHost));checkCudaErrors(cudaMemcpy(b.data(),bits,b.size()*4,cudaMemcpyDeviceToHost));FILE*f=fopen("reports/pathtracer-material-native.bin","wb");if(!f)return 2;fwrite(v.data(),4,v.size(),f);fclose(f);f=fopen("reports/pathtracer-material-native-bits.bin","wb");if(!f)return 3;fwrite(b.data(),4,b.size(),f);fclose(f);checkCudaErrors(cudaFree(out));checkCudaErrors(cudaFree(bits));printf("512 original material scatter cases; 5120 float outputs and 512 subsequent RNG draws\n");}
