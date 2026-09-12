#include <cuda_runtime.h>
#include <cstdio>
#include "volatile-compound.cu"
int main(){
 int *out=nullptr,ints[64];float *fout=nullptr,floats[32];
 if(cudaMalloc(&out,sizeof(ints))!=cudaSuccess||cudaMalloc(&fout,sizeof(floats))!=cudaSuccess)return 1;
 volatile_compound<<<1,32>>>(out,fout);
 if(cudaDeviceSynchronize()!=cudaSuccess)return 2;
 if(cudaMemcpy(ints,out,sizeof(ints),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(floats,fout,sizeof(floats),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
 cudaFree(out);cudaFree(fout);int errors=0;
 for(int i=0;i<32;i++){int lane=(i+1)%32;for(int row=0;row<2;row++)if(ints[i*2+row]!=((((row*100+lane+7)*3)>>1)^5))errors++;if(floats[i]!=(lane+1)/2.f)errors++;}
 std::printf("{\"mode\":\"native CUDA\",\"values\":96,\"errors\":%d}\n",errors);return errors?4:0;
}
