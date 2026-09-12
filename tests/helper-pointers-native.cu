// Original project regression harness, MIT.
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include "helper-pointers.cu"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(int threads:{32,128}){
 int n=threads*3;std::vector<float>input(n+1),other(n,999),out(n+17,-12345);
 std::vector<float4>vectors(n),copied(n+4,make_float4(-12345,-12345,-12345,-12345));
 for(int i=0;i<=n;i++)input[i]=i*0.125f;
 for(int i=0;i<n;i++)vectors[i]=make_float4(i*0.25f,-i*0.5f,2,1);
 float *a,*b,*o;float4 *v,*c;int *count,actual=0;
 CHECK(cudaMalloc(&a,input.size()*4));CHECK(cudaMalloc(&b,other.size()*4));CHECK(cudaMalloc(&o,out.size()*4));
 CHECK(cudaMalloc(&v,vectors.size()*16));CHECK(cudaMalloc(&c,copied.size()*16));CHECK(cudaMalloc(&count,4));
 CHECK(cudaMemcpy(a,input.data(),input.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,other.data(),other.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(o,out.data(),out.size()*4,cudaMemcpyHostToDevice));
 CHECK(cudaMemcpy(v,vectors.data(),vectors.size()*16,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(c,copied.data(),copied.size()*16,cudaMemcpyHostToDevice));CHECK(cudaMemset(count,0,4));
 testPointerHelpers<<<3,threads>>>(a,b,v,o,c,count);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 CHECK(cudaMemcpy(out.data(),o,out.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(copied.data(),c,copied.size()*16,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&actual,count,4,cudaMemcpyDeviceToHost));
 if(actual!=n||out[n]!=4)return 1;
 for(int i=0;i<n;i++)if(out[i]!=input[i]+input[i+1]||copied[i].x!=vectors[i].x||copied[i].y!=vectors[i].y||copied[i].z!=2||copied[i].w!=1)return 1;
 for(int i=n+1;i<n+17;i++)if(out[i]!=-12345)return 1;
 for(int i=n;i<n+4;i++)if(copied[i].x!=-12345||copied[i].y!=-12345||copied[i].z!=-12345||copied[i].w!=-12345)return 1;
 CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(o));CHECK(cudaFree(v));CHECK(cudaFree(c));CHECK(cudaFree(count));
 printf("threads=%d groups=3 PASS nested offsets, float4 copy, atomics, aliases, shared reuse and guards\n",threads);
}return 0;}
