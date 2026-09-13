// MIT native reference; padding bytes are deliberately not bool objects.
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdio>
#include "bool-storage.cu"
int main(){static_assert(sizeof(bool)==1,"CUDA bool byte ABI");unsigned n=1025;std::vector<unsigned char> input(n),output(1032,165),flag(4,165);std::vector<int> observed(n);for(unsigned i=0;i<n;i++)input[i]=(i%3)==1;
 bool *a,*b,*c;int* d;
 if(cudaMalloc(&a,n)!=cudaSuccess||cudaMalloc(&b,1032)!=cudaSuccess||cudaMalloc(&c,4)!=cudaSuccess||cudaMalloc(&d,n*4)!=cudaSuccess)return 1;
 cudaMemcpy(a,input.data(),n,cudaMemcpyHostToDevice);cudaMemcpy(b,output.data(),1032,cudaMemcpyHostToDevice);cudaMemcpy(c,flag.data(),4,cudaMemcpyHostToDevice);
 for(int repeat=0;repeat<8;repeat++)boolStorage<<<(n+127)/128,128>>>(a,b,c,d,n);
 if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
 if(cudaMemcpy(output.data(),b,1032,cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(flag.data(),c,4,cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(observed.data(),d,n*4,cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
 std::ofstream f("reports/bool-storage-native.bin",std::ios::binary);f.write((char*)output.data(),1032);f.write((char*)flag.data(),4);f.write((char*)observed.data(),n*4);cudaFree(a);cudaFree(b);cudaFree(c);cudaFree(d);return f?0:4;}
