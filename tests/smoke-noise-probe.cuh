// SPDX-License-Identifier: MIT
// Native and WebGPU probe around the unchanged NVIDIA noise3D helper.
__global__ void sampleSmokeNoise(const float4* coordinates,float4* output,cudaTextureObject_t texture,unsigned int count){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float4 p=coordinates[i];float3 value=noise3D(make_float3(p.x,p.y,p.z),texture);
 output[i]=make_float4(value,tex3D<float4>(texture,p.x,p.y,p.z).w);
}
