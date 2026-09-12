// MIT compiler validation probe.
__device__ float adjusted(float x, float scale = 2.0f, int bias = -3) { return x * scale + bias; }
template<class T> __device__ T addDefault(T x, T y = 4) { return x + y; }
template<> __device__ int addDefault<int>(int x, int y) { return x - y; }
template<class Tag,class R> __device__ R sampleDefault(cudaTextureObject_t tex,float x,float y,int component=0) { return tex2D<R>(tex,x,y)+(R)component; }
__device__ float nestedDefault(float x) { return adjusted(x); }
__device__ int chooseDefault(int x,int y=7) { return x+y; }
__device__ float chooseDefault(float x,float y=0.5f) { return x+y; }
__global__ void helperDefaults(const float* input,float* output,cudaTextureObject_t tex) {
 output[0]=adjusted(input[0]);
 output[1]=adjusted(input[0],3.0f);
 output[2]=adjusted(input[0],3.0f,5);
 output[3]=addDefault(input[0]);
 output[4]=addDefault<int>((int)input[0]);
 output[5]=sampleDefault<int,float>(tex,1.5f,0.5f);
 output[6]=sampleDefault<int,float>(tex,1.5f,0.5f,3);
 output[7]=nestedDefault(input[1]);
 output[8]=chooseDefault((int)input[0]);
 output[9]=chooseDefault(input[0]);
}
