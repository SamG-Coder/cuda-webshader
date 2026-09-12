#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include "../.local/raytracing-cuda/camera.h"
#include "pathtracer-camera-kernels.cuh"
int main(){unsigned int*bits;float*out;checkCudaErrors(cudaMalloc(&bits,512*9*4));checkCudaErrors(cudaMalloc(&out,512*23*4));check_camera<<<8,64>>>(bits,out);checkCudaErrors(cudaDeviceSynchronize());std::vector<unsigned int>b(512*9);std::vector<float>v(512*23);checkCudaErrors(cudaMemcpy(b.data(),bits,b.size()*4,cudaMemcpyDeviceToHost));checkCudaErrors(cudaMemcpy(v.data(),out,v.size()*4,cudaMemcpyDeviceToHost));FILE*f=fopen("reports/pathtracer-camera-native-bits.bin","wb");if(!f)return 2;fwrite(b.data(),4,b.size(),f);fclose(f);f=fopen("reports/pathtracer-camera-native.bin","wb");if(!f)return 3;fwrite(v.data(),4,v.size(),f);fclose(f);checkCudaErrors(cudaFree(bits));checkCudaErrors(cudaFree(out));printf("512 cameras; 4608 RNG integers and 11776 uniform/camera float outputs\n");}
