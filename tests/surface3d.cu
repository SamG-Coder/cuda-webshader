// SPDX-License-Identifier: MIT
__global__ void writeVolume(cudaSurfaceObject_t dst){unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;unsigned int z=blockIdx.z*blockDim.z+threadIdx.z;float value=(float)((z*8+y)*8+x)*0.125f-16.0f;surf3Dwrite(value,dst,x*4,y,z);}
