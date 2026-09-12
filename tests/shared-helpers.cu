// Original project regression fixture, MIT.
template<class T> __device__ T reverseTile(T value,cooperative_groups::thread_block team){
 __shared__ T tile[128];tile[threadIdx.x]=value;team.sync();
 T result=tile[blockDim.x-1u-threadIdx.x];team.sync();return result;
}
__device__ unsigned int countLanes(){
 __shared__ unsigned int count;if(threadIdx.x==0u)count=0u;__syncthreads();
 atomicAdd(&count,1u);__syncthreads();return count;
}
__device__ float outer(){
 cooperative_groups::thread_block team=cooperative_groups::this_thread_block();
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 float a=reverseTile((float)i*0.125f,team);unsigned int b=reverseTile(i,team);
 return a+(float)b+(float)countLanes()+(float)gridDim.x;
}
__global__ void testSharedHelpers(float* out){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;out[i]=outer();}
