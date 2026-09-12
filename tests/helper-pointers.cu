// Original project regression fixture, MIT.
template<class T> __device__ void copyRecord(const T* src,T* dst){dst[0]=src[0];}
__device__ float prior(const float* p){return p[-1];}
__device__ float nested(const float* p){const float* q=p+1;return prior(q);}
__device__ void increment(int* p){atomicAdd(&p[0],1);}
__device__ void nestedIncrement(int* p){increment(p);}
__device__ float carry(const float* p,int write){
 __shared__ float tile[128];
 if(write!=0)tile[threadIdx.x]=p[threadIdx.x];
 __syncthreads();float value=tile[threadIdx.x];__syncthreads();return value;
}
__device__ void aliasWrite(float* a,float* b){a[0]=3.0f;b[0]=a[0]+1.0f;}
__global__ void testPointerHelpers(const float* input,const float* other,const float4* vectors,float* out,float4* copied,int* counts){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 float saved=carry(input+blockIdx.x*blockDim.x,1);
 out[i]=carry(other+blockIdx.x*blockDim.x,0)+nested(&input[i+1u]);
 copyRecord(&vectors[i],&copied[i]);
 nestedIncrement(counts);
 if(i==0u)aliasWrite(&out[gridDim.x*blockDim.x],&out[gridDim.x*blockDim.x]);
}
