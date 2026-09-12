// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#ifndef PICK
#define PICK 0
#endif
#if PICK
__device__ int selected(){return 19;}
#else
__device__ int selected(){return 7;}
#endif
#if 1
__global__ void conditionalKernel(int* out){out[0]=selected();}
#else
__global__ void conditionalKernel(int* out, const missingType){invalid_inactive_branch();}
#endif
#if 0
#if 1
__device__ void disabled(){unsupported_inactive_call();}
#endif
#endif
