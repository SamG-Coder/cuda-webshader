// MIT. Native CUDA probe uses NVIDIA's helper_math overloads.
#include <cuda_runtime.h>
#include <helper_math.h>
#include <cstdio>
#include <cmath>
#include "ray-vector-math.cu"
int main(){float4* device;auto e=cudaMalloc(&device,4*sizeof(float4));if(e!=cudaSuccess)return 2;float4* input;const float inputValues[]={3,0,4,1};e=cudaMalloc(&input,16);if(e!=cudaSuccess)return 2;e=cudaMemcpy(input,inputValues,16,cudaMemcpyHostToDevice);if(e!=cudaSuccess)return 2;rayMath<<<1,1>>>(input,device);float result[16];e=cudaMemcpy(result,device,sizeof(result),cudaMemcpyDeviceToHost);cudaFree(device);cudaFree(input);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}const float expected[]={.6f,0,.8f,1,.5f,0,.5f,1,.6f,.5f,.8f,2.4f,1.1f,1.5f,3.3f,1};float maxError=0;for(int i=0;i<16;i++){float error=fabsf(result[i]-expected[i]);maxError=fmaxf(maxError,error);if(!std::isfinite(result[i])||error>2e-6f){printf("FAIL component=%d\n",i);return 1;}}printf("PASS native CUDA helper_math ray operations; max error %.9g\n",maxError);return 0;}
