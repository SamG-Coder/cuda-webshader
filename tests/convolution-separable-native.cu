// Original project reference harness, MIT. Included NVIDIA code retains its BSD notice.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include "convolution-separable.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){float coefficients[17];for(int i=0;i<17;i++)coefficients[i]=(i+1)/256.0f;CHECK(cudaMemcpyToSymbol(c_Kernel,coefficients,sizeof(coefficients)));
 for(auto shape:{std::array<int,3>{128,64,128},{256,128,272}}){int w=shape[0],h=shape[1],pitch=shape[2],count=pitch*h+16;std::vector<float>input(count,-12345),expected(count,-12345),actual(count,-12345);for(int i=0;i<pitch*h;i++)if(i%pitch<w)input[i]=(i%29-14)*0.125f;
 float *a,*b;size_t bytes=count*4;CHECK(cudaMalloc(&a,bytes));CHECK(cudaMalloc(&b,bytes));CHECK(cudaMemcpy(a,input.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,actual.data(),bytes,cudaMemcpyHostToDevice));
 for(int axis=0;axis<2;axis++){for(int y=0;y<h;y++)for(int x=0;x<w;x++){double sum=0;for(int j=-8;j<=8;j++){int sx=x+(axis==0?j:0),sy=y+(axis==1?j:0);if(sx>=0&&sx<w&&sy>=0&&sy<h)sum+=double(input[sy*pitch+sx])*coefficients[8-j];}expected[y*pitch+x]=float(sum);}
 if(axis==0)convolutionRowsKernel<<<dim3(w/128,h/4),dim3(16,4)>>>(b,a,w,h,pitch);else convolutionColumnsKernel<<<dim3(w/16,h/64),dim3(16,8)>>>(b,a,w,h,pitch);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(actual.data(),b,bytes,cudaMemcpyDeviceToHost));for(int i=0;i<count;i++)if(!std::isfinite(actual[i])||std::abs(actual[i]-expected[i])>3e-5f){printf("FAIL axis=%d index=%d\n",axis,i);return 1;}input=expected;auto tmp=a;a=b;b=tmp;
 }CHECK(cudaFree(a));CHECK(cudaFree(b));printf("%dx%d pitch=%d PASS original row/column pipeline, independent reference and padding guards\n",w,h,pitch);
 }return 0;
}
