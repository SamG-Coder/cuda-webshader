// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "size-values-kernel.cuh"
#define CHECK(x) do{if((x)!=cudaSuccess)return 2;}while(0)
int main(){
 unsigned*words;float*values;CHECK(cudaMalloc(&words,20));CHECK(cudaMalloc(&values,8));
 unsigned pitches[]={192,0xffffffffu,12345,0,0xffffffffu};int rows[]={7,2,-1,0,2147483647};
 std::vector<unsigned> w(25);std::vector<float> v(10);
 for(int i=0;i<5;i++){sizeValues<<<1,1>>>(words,values,pitches[i],rows[i]);CHECK(cudaMemcpy(w.data()+i*5,words,20,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(v.data()+i*2,values,8,cudaMemcpyDeviceToHost));printf("Captured size_t: pitch=%u row=%d\n",pitches[i],rows[i]);}
 FILE*f=fopen("reports/size-values-words.bin","wb");if(!f)return 2;fwrite(w.data(),4,w.size(),f);fclose(f);f=fopen("reports/size-values-floats.bin","wb");if(!f)return 2;fwrite(v.data(),4,v.size(),f);fclose(f);CHECK(cudaFree(words));CHECK(cudaFree(values));return 0;
}
