// SPDX-License-Identifier: MIT
__device__ unsigned int readByte(const unsigned char* data, int offset) { return data[offset]; }
__global__ void vectorBytes(const unsigned char* input, unsigned int* output, float4* positions, unsigned int n, uint3 mask, uint3 shift, float3 spacing, int3 delta, float2 offset, float4 weights, uint2 pair, int4 edge) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){
 uint3 p=make_uint3(i&mask.x,(i>>shift.y)&mask.y,(i>>shift.z)&mask.z);
 output[i*4]=readByte(input+1,i);output[i*4+1]=readByte(input,i+2);
 output[i*4+2]=pair.x+pair.y+shift.x+mask.z;output[i*4+3]=(unsigned int)(delta.x+delta.y+delta.z+edge.x+edge.y+edge.z+edge.w);
 positions[i]=make_float4((float)p.x*spacing.x+offset.x+weights.x,(float)p.y*spacing.y+offset.y+weights.y,(float)p.z*spacing.z+weights.z,weights.w);
 }
}
