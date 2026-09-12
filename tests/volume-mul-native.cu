// MIT validation harness. Included NVIDIA helpers retain their BSD notice.
#include <cuda_runtime.h>
#include <helper_math.h>
#include <cstdio>
#include "volume-mul.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){const float matrix[]={2,0,0,10,0,3,0,20,0,0,4,30},values[]={1,2,3,1},expected[]={2,6,12,0,12,26,42,1};CHECK(cudaMemcpyToSymbol(c_invViewMatrix,matrix,sizeof(matrix)));float4 *input,*output;CHECK(cudaMalloc(&input,16));CHECK(cudaMalloc(&output,32));CHECK(cudaMemcpy(input,values,16,cudaMemcpyHostToDevice));transformProbe<<<1,1>>>(input,output);CHECK(cudaGetLastError());float result[8];CHECK(cudaMemcpy(result,output,32,cudaMemcpyDeviceToHost));for(int i=0;i<8;i++)if(result[i]!=expected[i]){printf("FAIL component=%d\n",i);return 1;}CHECK(cudaFree(input));CHECK(cudaFree(output));printf("PASS original NVIDIA mul overloads: direction and translated point, const matrix and temporary references\n");return 0;}
