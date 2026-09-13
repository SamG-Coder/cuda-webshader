#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include "chrono-integration.cu"
#include "chrono-eos-probe.cuh"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){const unsigned n=30327;ChFsiParamsSPH p;std::ifstream params(".local/chrono-params.bin",std::ios::binary);params.read((char*)&p,sizeof(p));if(!params)return 2;check(cudaMemcpyToSymbol(paramsD,&p,sizeof(p)));std::vector<Real4> input(n);std::ifstream file("reports/chrono-rk2-half-native.bin",std::ios::binary);file.seekg(n*28);file.read((char*)input.data(),n*16);if(!file)return 3;Real4* in;float* out;check(cudaMalloc(&in,n*16));check(cudaMalloc(&out,n*16));check(cudaMemcpy(in,input.data(),n*16,cudaMemcpyHostToDevice));eosProbe<<<(n+127)/128,128>>>(in,out,n);check(cudaGetLastError());std::vector<float> output(n*4);check(cudaMemcpy(output.data(),out,n*16,cudaMemcpyDeviceToHost));std::ofstream dst("reports/chrono-eos-native.bin",std::ios::binary);dst.write((char*)output.data(),n*16);check(cudaFree(in));check(cudaFree(out));return dst?0:4;}
