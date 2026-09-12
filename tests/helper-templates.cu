// Compiler feature regression fixture, MIT.
template<typename T> __device__ T square(T x) { return x*x; }
template<class T> __device__ T nested(T x) { return square(x)+x; }
template<int N> __device__ float repeat(float x) {float result=0.0f;for(int j=0;j<N;j++)result+=x;return result;}
template<typename T> __device__ void exchange(T& a,T& b) {T t=a;a=b;b=t;}
template<class T> __device__ T identity(T x) {return x;}
__global__ void helperTemplates(float* out,unsigned int* bits,unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){float x=((int)(i%13u)-6)*0.25f;float a=x;float b=2.0f;exchange(a,b);
 float4 v=identity(make_float4(x,x+1.0f,x+2.0f,x+3.0f));
 out[i*4u]=nested(x);out[i*4u+1u]=repeat<3>(x);out[i*4u+2u]=a;out[i*4u+3u]=v.y+rsqrt_T(4.0f);
 bits[i]=nested(0x80000000u+i);
 }
}
