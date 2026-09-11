// Syntax checking only. This header is NOT a CUDA emulator and does not execute kernels.
#pragma once
#include <cmath>
#include <cstdint>
#define __global__
#define __device__
#define __shared__
#define __host__
#define __forceinline__ inline
struct dim3_stub { unsigned int x=0,y=0,z=0; };
extern dim3_stub threadIdx,blockIdx,blockDim,gridDim;
struct float2 {float x,y;};
struct float3 {float x,y,z;};
struct alignas(16) float4 {float x,y,z,w;};
inline float2 make_float2(float x,float y){return {x,y};}
inline float3 make_float3(float x,float y,float z){return {x,y,z};}
inline float4 make_float4(float x,float y,float z,float w){return {x,y,z,w};}
inline float rsqrtf(float v){return 1.0f/std::sqrt(v);}
inline void __syncthreads(){}
template<class T> T atomicAdd(T*,T);
template<class T> T atomicMin(T*,T);
template<class T> T atomicMax(T*,T);
template<class T> T atomicExch(T*,T);
