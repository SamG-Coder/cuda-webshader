// SPDX-License-Identifier: MIT
template<class T> __device__ uint typeBytes(){return sizeof(T);}
__global__ void sizeProbe(const uint* input,uint* output){uint i=threadIdx.x;if(i<5){uint x=input[i];output[i*4]=(uint)(x*sizeof(float4));output[i*4+1]=(x*sizeof(float4))>4294967295u;output[i*4+2]=typeBytes<float3>();output[i*4+3]=sizeof(cudaExtent);}}
__global__ void byteWrite(cudaSurfaceObject_t dst){uint x=blockIdx.x*blockDim.x+threadIdx.x;uint y=blockIdx.y*blockDim.y+threadIdx.y;uint z=blockIdx.z*blockDim.z+threadIdx.z;unsigned char value=(unsigned char)(x+y*8+z*64);surf3Dwrite(value,dst,x*sizeof(unsigned char),y,z);}
__global__ void byteRead(cudaTextureObject_t src,uint* output){uint x=blockIdx.x*blockDim.x+threadIdx.x;uint y=blockIdx.y*blockDim.y+threadIdx.y;uint z=blockIdx.z*blockDim.z+threadIdx.z;output[x+y*8+z*64]=(uint)(tex3D<float>(src,((float)x+.5f)/8.0f,((float)y+.5f)/8.0f,((float)z+.5f)/4.0f)*255.0f+.5f);}
