// SPDX-License-Identifier: MIT
__device__ void writeRed(uchar4* p, uint i){p[i].x=250;p[i].x+=10;p[i].x++;}
__global__ void componentStores(uchar4* out,uint n){uint lane=blockIdx.x*blockDim.x+threadIdx.x;uint i=lane/4;if(i<n){uint channel=lane%4;if(channel==0)writeRed(out,i);if(channel==1)out[i].y=i+33;if(channel==2){out[i].z=2;out[i].z--;}if(channel==3)out[i].w=128+i;}}
__global__ void copyColours(const uchar4* input,uchar4* output,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=input[i];}

__device__ int nextPackedIndex(int& i){return i++;}
__global__ void orderedStores(uchar4* out){int i=0;out[i].x=++i;out[nextPackedIndex(i)].y+=2;}
