// MIT probe for CUDA local struct and constant matrix value semantics.
#include <cuda_runtime.h>
#include <helper_math.h>
#include <cstdio>
#include "local-structs.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){float inputValues[]={1,2,3,1,4,5,6,1},matrixValues[12];for(int i=0;i<12;i++)matrixValues[i]=float(i+1);float3x4 matrix;memcpy(&matrix,matrixValues,sizeof(matrix));CHECK(cudaMemcpyToSymbol(camera,&matrix,sizeof(matrix)));float4 *input,*output;CHECK(cudaMalloc(&input,sizeof(inputValues)));CHECK(cudaMalloc(&output,64));CHECK(cudaMemcpy(input,inputValues,sizeof(inputValues),cudaMemcpyHostToDevice));
 for(unsigned row=0;row<3;row++){structProbe<<<1,1>>>(input,output,row);CHECK(cudaGetLastError());float actual[16];CHECK(cudaMemcpy(actual,output,64,cudaMemcpyDeviceToHost));float expected[]={5,7,9,1,1,2,3,1,float(row*4+1),float(row*4+2),float(row*4+3),float(row*4+4),1,2,3,4};if(row==0)expected[8]=99;for(int i=0;i<16;i++)if(actual[i]!=expected[i]){printf("FAIL row=%u component=%d\n",row,i);return 1;}printf("row=%u PASS native struct copy, helper return, dynamic matrix row and unchanged constant\n",row);}CHECK(cudaFree(input));CHECK(cudaFree(output));return 0;}
