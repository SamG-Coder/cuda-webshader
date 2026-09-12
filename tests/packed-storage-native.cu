// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
typedef unsigned int uint;
#include "packed-storage.cu"
int main(){constexpr unsigned n=257;std::vector<unsigned> actual(n+16,0xdeadbeef);uchar4* out;cudaMalloc(&out,actual.size()*4);cudaMemcpy(out,actual.data(),actual.size()*4,cudaMemcpyHostToDevice);componentStores<<<(n*4+63)/64,64>>>(out,n);auto e=cudaMemcpy(actual.data(),out,actual.size()*4,cudaMemcpyDeviceToHost);cudaFree(out);if(e!=cudaSuccess)return 2;for(unsigned i=0;i<n+16;i++){unsigned expected=i<n?(5u|((i+33)&255)<<8|1u<<16|((128+i)&255)<<24):0xdeadbeef;if(actual[i]!=expected){printf("FAIL %u %08x != %08x\n",i,actual[i],expected);return 1;}}std::vector<unsigned> ordered(3,0x04030201);cudaMalloc(&out,12);cudaMemcpy(out,ordered.data(),12,cudaMemcpyHostToDevice);orderedStores<<<1,1>>>(out);e=cudaMemcpy(ordered.data(),out,12,cudaMemcpyDeviceToHost);cudaFree(out);if(e!=cudaSuccess||ordered[0]!=0x04030201||ordered[1]!=0x04030401||ordered[2]!=0x04030201)return 3;printf("PASS C++17 packed destination evaluation ordering\n");printf("PASS %u packed records: four independent byte writers, helper stores, wraparound, increments and unchanged guards\n",n);}
