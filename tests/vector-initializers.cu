// Original project regression fixture, MIT.
template<class T> struct InitializerVector {typedef float3 Type;};
template<class T> __device__ T initializedSum(T x){
 typename InitializerVector<T>::Type acc={0.0f,0.0f,0.0f};
 typename InitializerVector<T>::Type values={x,2.0f,3.0f};
 acc.x=values.x+values.y+values.z;return acc.x;
}
__global__ void testVectorInitializers(float4* out,unsigned int n){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){float4 empty={},partial={initializedSum((float)i),},full={1.0f,2.0f,3.0f,4.0f};
 out[i*3u]=empty;out[i*3u+1u]=partial;out[i*3u+2u]=full;}
}
