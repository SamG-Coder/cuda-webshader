// Original project regression fixture, MIT; SharedMemory is NVIDIA's original wrapper.
template<class T> __device__ void reverseShared(const T* input,T* output){
 T* tile=SharedMemory<T>();
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 tile[threadIdx.x]=input[i];__syncthreads();
 output[i]=tile[blockDim.x-threadIdx.x-1u];__syncthreads();
}
__global__ void testSharedWrapper(const float4* input,float4* output){reverseShared(input,output);}
