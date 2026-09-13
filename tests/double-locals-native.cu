#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include "double-locals.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){const unsigned n=129;std::vector<float> input(n),output(n*3);std::vector<Pair> pairs(n);for(unsigned i=0;i<n;i++){input[i]=(i+1)*0.125f;pairs[i]={float(i),-float(i)};}
 input[n-1]=1e20f;
 float *a,*b;Pair* p;check(cudaMalloc(&a,n*4));check(cudaMalloc(&b,n*12));check(cudaMalloc(&p,n*8));check(cudaMemcpy(a,input.data(),n*4,cudaMemcpyHostToDevice));check(cudaMemcpy(p,pairs.data(),n*8,cudaMemcpyHostToDevice));
 doubleLocals<<<2,128>>>(a,b,n);check(cudaGetLastError());recordRefs<<<2,128>>>(p,n);check(cudaGetLastError());check(cudaMemcpy(output.data(),b,n*12,cudaMemcpyDeviceToHost));check(cudaMemcpy(pairs.data(),p,n*8,cudaMemcpyDeviceToHost));
 std::ofstream file("reports/double-locals-native.bin",std::ios::binary);file.write((char*)output.data(),n*12);file.write((char*)pairs.data(),n*8);check(cudaFree(a));check(cudaFree(b));check(cudaFree(p));return file?0:2;}
