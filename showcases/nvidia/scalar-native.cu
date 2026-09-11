// Original verification harness: MIT. Included NVIDIA kernel: BSD-3-Clause.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <cstdint>
#include "kernels/20.cu"
#define CHECK(call) do {cudaError_t error=(call); if(error!=cudaSuccess){printf("%s\n",cudaGetErrorString(error));return 2;}}while(0)
__global__ void multiply24(int *s,unsigned int *u,const int *a,const int *b,int n){int i=threadIdx.x;if(i<n){s[i]=__mul24(a[i],b[i]);u[i]=__umul24((unsigned int)a[i],(unsigned int)b[i]);}}
int main(){
 const int cases[][4]={{1,1,1,32},{7,33,3,128},{17,1537,4,128},{5,4096,2,256}};
 for(const auto& test:cases){int vectorN=test[0],elementN=test[1],n=vectorN*elementN;std::vector<float>a(n+16),b(n+16),c(vectorN+16,-12345),expected=c;for(int i=0;i<n+16;i++){a[i]=(i%23-11)/8.0f;b[i]=(i%17-8)/8.0f;}for(int v=0;v<vectorN;v++){double sum=0;for(int i=0;i<elementN;i++)sum+=a[v*elementN+i]*b[v*elementN+i];expected[v]=(float)sum;}
  float *ga,*gb,*gc;size_t bytes=a.size()*sizeof(float),outBytes=c.size()*sizeof(float);CHECK(cudaMalloc(&ga,bytes));CHECK(cudaMalloc(&gb,bytes));CHECK(cudaMalloc(&gc,outBytes));CHECK(cudaMemcpy(ga,a.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(gb,b.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(gc,c.data(),outBytes,cudaMemcpyHostToDevice));scalarProdGPU<<<test[2],test[3]>>>(gc,ga,gb,vectorN,elementN);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(c.data(),gc,outBytes,cudaMemcpyDeviceToHost));if(c!=expected){printf("FAIL scalar product\n");return 1;}CHECK(cudaFree(ga));CHECK(cudaFree(gb));CHECK(cudaFree(gc));printf("scalarProdGPU vectors=%d elements=%d blocks=%d threads=%d PASS (16 guards)\n",vectorN,elementN,test[2],test[3]);
 }
 const int values[]={0,1,-1,8388607,8388608,16777215,16777216,2147483647,INT32_MIN,305419896,-305419896};std::vector<int>a,b;for(int x:values)for(int y:values){a.push_back(x);b.push_back(y);}int n=(int)a.size();std::vector<int>s(n+16,-12345);std::vector<unsigned int>u(n+16,12345);int *ga,*gb,*gs;unsigned int *gu;size_t bytes=n*sizeof(int),outBytes=s.size()*sizeof(int);CHECK(cudaMalloc(&ga,bytes));CHECK(cudaMalloc(&gb,bytes));CHECK(cudaMalloc(&gs,outBytes));CHECK(cudaMalloc(&gu,outBytes));CHECK(cudaMemcpy(ga,a.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(gb,b.data(),bytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(gs,s.data(),outBytes,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(gu,u.data(),outBytes,cudaMemcpyHostToDevice));multiply24<<<1,128>>>(gs,gu,ga,gb,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(s.data(),gs,outBytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(u.data(),gu,outBytes,cudaMemcpyDeviceToHost));
 for(int i=0;i<n;i++){int64_t x=(uint32_t)a[i]&0xffffff,y=(uint32_t)b[i]&0xffffff;uint32_t eu=(uint32_t)(x*y);if(x>=0x800000)x-=0x1000000;if(y>=0x800000)y-=0x1000000;if((uint32_t)s[i]!=(uint32_t)(x*y)||u[i]!=eu){printf("FAIL mul24 pair %d\n",i);return 1;}}for(int i=n;i<n+16;i++)if(s[i]!=-12345||u[i]!=12345){printf("FAIL guard\n");return 1;}CHECK(cudaFree(ga));CHECK(cudaFree(gb));CHECK(cudaFree(gs));CHECK(cudaFree(gu));printf("mul24/umul24: %d pairs PASS (sign, truncation, overflow, guards)\n",n);
}
