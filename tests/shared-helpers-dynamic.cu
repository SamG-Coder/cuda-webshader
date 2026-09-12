// Original project regression fixture, MIT.
__device__ float reverseDynamic(float value){extern __shared__ float tile[];tile[threadIdx.x]=value;__syncthreads();float result=tile[blockDim.x-1u-threadIdx.x];__syncthreads();return result;}
__global__ void testDynamicHelper(float* out){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;out[i]=reverseDynamic((float)i);}
