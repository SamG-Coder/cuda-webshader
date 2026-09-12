// MIT native validation host.
#include <cuda_runtime.h>
#include <cstdio>
#include "expression-macros.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int data[]={16777217,-8388607,2,3},expected[]={5,-16777213,11,25,17},result[5],*input,*output;CHECK(cudaMalloc(&input,sizeof(data)));CHECK(cudaMalloc(&output,sizeof(result)));CHECK(cudaMemcpy(input,data,sizeof(data),cudaMemcpyHostToDevice));expressionMacros<<<1,1>>>(input,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(result,output,sizeof(result),cudaMemcpyDeviceToHost));for(int i=0;i<5;i++)if(result[i]!=expected[i]){printf("FAIL component=%d\n",i);return 1;}CHECK(cudaFree(input));CHECK(cudaFree(output));printf("PASS native expression macros: signed 24-bit IMAD, argument precedence, repeated parameters and dependent numeric constants\n");return 0;}
