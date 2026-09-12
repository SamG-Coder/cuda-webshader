// MIT native validation harness for the unchanged Chrono enum declarations.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "chrono-enums.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){int* output;check(cudaMalloc(&output,54*4));printf("[");
 for(int i=0;i<4;i++){Flags value{bool(i&1),bool(i&2)};check(cudaMemcpyToSymbol(flags,&value,sizeof(value)));chronoEnumValues<<<1,1>>>(output);check(cudaGetLastError());check(cudaDeviceSynchronize());int values[54];check(cudaMemcpy(values,output,sizeof(values),cudaMemcpyDeviceToHost));printf("%s[",i?",":"");for(int j=0;j<54;j++)printf("%s%d",j?",":"",values[j]);printf("]");}
 printf("]\n");check(cudaFree(output));return 0;}
