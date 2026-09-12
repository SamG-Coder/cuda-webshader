// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdint>
#include "float64-expression-kernel.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}}while(0)
__device__ uint2 words(double x){auto bits=__double_as_longlong(x);return make_uint2((unsigned int)bits,(unsigned int)((unsigned long long)bits>>32));}
__global__ void evaluate(const uint2* input,uint2* output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint2 x=input[i*2],y=input[i*2+1];
 double a=__longlong_as_double(((unsigned long long)x.y<<32)|x.x),b=__longlong_as_double(((unsigned long long)y.y<<32)|y.x);
 output[i*10]=words(a+b);output[i*10+1]=words(a-b);output[i*10+2]=words(a*b);output[i*10+3]=words(a/b);
 output[i*10+4]=make_uint2((a<b)|((a==b)<<1)|((a>b)<<2),0);
 output[i*10+5]=make_uint2(__float_as_uint((float)a),__float_as_uint((float)b));
 output[i*10+6]=words((double)__uint_as_float(x.x));
 output[i*10+7]=words((double)x.x);output[i*10+8]=words((double)(int)x.x);
 output[i*10+9]=make_uint2(__float_as_uint((float)(((unsigned long long)x.y<<32)|x.x)),__float_as_uint((float)(((unsigned long long)y.y<<32)|y.x)));
}
int main(){
 const uint64_t edges[]={0,0x8000000000000000ULL,1,0x8000000000000001ULL,0xfffffffffffffULL,0x10000000000000ULL,0x10000000000001ULL,0x3ff0000000000000ULL,0xbff0000000000000ULL,0x3ff0000000000001ULL,0x3fefffffffffffffULL,0x7fefffffffffffffULL,0xffefffffffffffffULL,0x7ff0000000000000ULL,0xfff0000000000000ULL,0x7ff8000000000001ULL,0x3e60000000000000ULL,0x3810000000000000ULL,0x3690000000000000ULL,0x3fe0000000000000ULL,0x4000000000000000ULL,0x3ca0000000000000ULL,0x3c90000000000000ULL,0x3ff0000010000000ULL,0x3ff0000030000000ULL};
 std::vector<uint2> input;auto push=[&](uint64_t v){input.push_back(make_uint2((uint32_t)v,(uint32_t)(v>>32)));};
 for(auto a:edges)for(auto b:edges){push(a);push(b);}
 uint64_t state=0x13579bdf2468ace1ULL;
 for(int i=0;i<8192;i++){state=state*6364136223846793005ULL+1442695040888963407ULL;push(state);state=state*6364136223846793005ULL+1442695040888963407ULL;push(state);}
 int count=input.size()/2;std::vector<uint2> output(count*10);uint2 *di,*dout;CHECK(cudaMalloc(&di,input.size()*sizeof(uint2)));CHECK(cudaMalloc(&dout,output.size()*sizeof(uint2)));
 CHECK(cudaMemcpy(di,input.data(),input.size()*sizeof(uint2),cudaMemcpyHostToDevice));evaluate<<<(count+127)/128,128>>>(di,dout,count);CHECK(cudaGetLastError());CHECK(cudaMemcpy(output.data(),dout,output.size()*sizeof(uint2),cudaMemcpyDeviceToHost));
 for(int i=0;i<2;i++){FILE* f=fopen(i?"reports/float64-native.bin":"reports/float64-input.bin","wb");if(!f)return 2;auto& data=i?output:input;fwrite(data.data(),sizeof(uint2),data.size(),f);fclose(f);}
 printf("Captured %d binary64 input pairs: 10 results per pair, native double arithmetic and conversions.\n",count);
 std::vector<float> a(count),b(count),results(count*6);for(int i=0;i<count;i++){auto x=input[i*2].x,y=input[i*2+1].x;memcpy(&a[i],&x,4);memcpy(&b[i],&y,4);}
 float *da,*db;CHECK(cudaMalloc(&da,count*4));CHECK(cudaMalloc(&db,count*4));CHECK(cudaMemcpy(da,a.data(),count*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(db,b.data(),count*4,cudaMemcpyHostToDevice));
 expressions<<<(count+127)/128,128>>>(da,db,(float*)dout,count);CHECK(cudaGetLastError());CHECK(cudaMemcpy(results.data(),dout,count*24,cudaMemcpyDeviceToHost));
 FILE* f=fopen("reports/float64-expressions-native.bin","wb");if(!f)return 2;fwrite(results.data(),4,results.size(),f);fclose(f);
 CHECK(cudaFree(da));CHECK(cudaFree(db));CHECK(cudaFree(di));CHECK(cudaFree(dout));return 0;
}
