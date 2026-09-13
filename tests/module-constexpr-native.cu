#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include "module-constexpr.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){float* output;bool* flags;check(cudaMalloc(&output,65*8*4));check(cudaMalloc(&flags,68));check(cudaMemset(flags,0xa5,68));
 constants<<<2,64>>>(output,flags);check(cudaGetLastError());check(cudaDeviceSynchronize());
 std::vector<unsigned char> bytes(65*8*4+68);check(cudaMemcpy(bytes.data(),output,65*8*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(bytes.data()+65*8*4,flags,68,cudaMemcpyDeviceToHost));
 std::ofstream file("reports/module-constexpr-native.bin",std::ios::binary);file.write((char*)bytes.data(),bytes.size());check(cudaFree(output));check(cudaFree(flags));return file?0:2;}
