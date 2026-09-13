#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
#include <limits>
#include "integer-powers.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){std::vector<float> input;const float bases[]={-17,-2,-1.003875f,-1,-.5f,-.125f,-0.f,0.f,.001f,.125f,.5f,1.f,1.00001f,1.003875f,2.f,17.f,std::numeric_limits<float>::min(),std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
 for(float b:bases)for(float e:{-64.f,-7.f,-2.f,-1.f,0.f,1.f,2.f,7.f,16.f,32.f,48.f,64.f}){input.push_back(b);input.push_back(e);}
 const unsigned n=input.size()/2;float *a,*b;check(cudaMalloc(&a,input.size()*4));check(cudaMalloc(&b,n*4));check(cudaMemcpy(a,input.data(),input.size()*4,cudaMemcpyHostToDevice));integerPowers<<<(n+63)/64,64>>>(a,b,n);check(cudaGetLastError());std::vector<float> output(n);check(cudaMemcpy(output.data(),b,n*4,cudaMemcpyDeviceToHost));
 std::ofstream src("reports/integer-powers-input.bin",std::ios::binary),dst("reports/integer-powers-native.bin",std::ios::binary);src.write((char*)input.data(),input.size()*4);dst.write((char*)output.data(),n*4);check(cudaFree(a));check(cudaFree(b));return src&&dst?0:2;}
