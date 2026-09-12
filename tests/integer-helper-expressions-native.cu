// MIT validation host.
#include <cuda_runtime.h>
#include <cstdio>
#include "integer-helper-expressions.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){int data[]={3,-2},expected[]={187,0,7,-2,(-2147483647-1),-3},result[6],*input,*output;CHECK(cudaMalloc(&input,sizeof(data)));CHECK(cudaMalloc(&output,sizeof(result)));CHECK(cudaMemcpy(input,data,sizeof(data),cudaMemcpyHostToDevice));integerTemplates<<<1,1>>>(input,output);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(result,output,sizeof(result),cudaMemcpyDeviceToHost));for(int i=0;i<6;i++)if(result[i]!=expected[i]){printf("FAIL component=%d actual=%d expected=%d\n",i,result[i],expected[i]);return 1;}CHECK(cudaFree(input));CHECK(cudaFree(output));printf("PASS native integer helper expressions: 187, 0, 7, -2, INT_MIN, -3; descending specialization, negative base and nested arithmetic\n");return 0;}
