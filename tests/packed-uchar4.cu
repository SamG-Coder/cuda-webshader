// MIT byte-layout validation probe.
__device__ int byteTag(uchar x) { return 7; }
__device__ int byteTag(int x) { return 13; }
__device__ int readByte(const uchar& x) { return x; }
__device__ void adjustByte(uchar& x) { x += 10; }
template<class T> __device__ T copyPacked(T x) { return x; }
__global__ void packedBytes(const float* values,uchar4* output,int* checks,unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){
  uchar4 p=make_uchar4(values[i],(int)i-130,i*17u,255u);
  uchar4 q=copyPacked(p);q.x+=250;q.y--;q.z=(uchar)(q.z+257);q.w++;
  uchar v=250;adjustByte(v);v++;v--;
  unsigned char c=0;
  checks[i]=(c-1)*10000+byteTag(p.x)*1000+readByte(p.x)+byteTag(min(p.x,p.y))*100+v;
  output[i*2]=p;output[i*2+1]=q;
 }
}
