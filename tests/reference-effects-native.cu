// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "reference-effects.cu"
int main(){float* out;float actual[13];if(cudaMalloc(&out,sizeof(actual))!=cudaSuccess)return 2;referenceEffects<<<1,1>>>(out);auto e=cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost);cudaFree(out);if(e!=cudaSuccess)return 2;const float expected[]={1.25f,3,3,1,3,1,1,3,12,4,4.125f,.1f,-0.f};for(int i=0;i<13;i++)if(actual[i]!=expected[i]||std::signbit(actual[i])!=std::signbit(expected[i]))return 1;printf("PASS reference chains, short-circuit updates, const uchar4 references and explicit float casts\n");for(float v:actual)printf("%.9g ",v);printf("\n");}
