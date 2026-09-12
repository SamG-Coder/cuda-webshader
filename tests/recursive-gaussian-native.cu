// Project reference harness, MIT. Included NVIDIA kernels retain their BSD notice.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <helper_math.h>
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include "recursive-gaussian.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){const double alpha=1.695/2,e=exp(-alpha),e2=e*e,k=(1-e)*(1-e)/(1+2*alpha*e-e2);const float a0=k,a1=k*(alpha-1)*e,a2=k*(alpha+1)*e,a3=-k*e2,b1=-2*e,b2=e2,cp=(k+k*(alpha-1)*e)/(1-2*e+e2),cn=(k*(alpha+1)*e-k*e2)/(1-2*e+e2);
 for(auto shape:{std::array<int,2>{17,9},{32,16},{128,64}}){int w=shape[0],h=shape[1],count=w*h;std::vector<unsigned> expected(count+16,0xdeadbeef),actual(count+16,0xdeadbeef);for(int i=0;i<count;i++)expected[i]=(i*13%256)|((i*29%256)<<8)|((i*7%256)<<16)|0xff000000u;unsigned *input,*output;CHECK(cudaMalloc(&input,(count+16)*4));CHECK(cudaMalloc(&output,(count+16)*4));CHECK(cudaMemcpy(input,expected.data(),(count+16)*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(output,actual.data(),(count+16)*4,cudaMemcpyHostToDevice));int maxError=0;
 for(int axis=0;axis<2;axis++){std::vector<unsigned> filtered(count+16,0xdeadbeef);for(int x=0;x<w;x++){std::vector<std::array<double,4>>forward(h),reverse(h);for(int c=0;c<4;c++){auto at=[&](int y){return double((expected[y*w+x]>>(8*c))&255)/255;};double prev=at(0),last=cp*prev,older=last;for(int y=0;y<h;y++){double value=a0*at(y)+a1*prev-b1*last-b2*older;forward[y][c]=int(fmax(0.,fmin(1.,value))*255)/255.;prev=at(y);older=last;last=value;}double next=at(h-1),nextInput2=next;last=cn*next;older=last;for(int y=h-1;y>=0;y--){double value=a2*next+a3*nextInput2-b1*last-b2*older;reverse[y][c]=value;nextInput2=next;next=at(y);older=last;last=value;}}for(int y=0;y<h;y++){unsigned value=0;for(int c=0;c<4;c++)value|=unsigned(fmax(0.,fmin(1.,forward[y][c]+reverse[y][c]))*255)<<(8*c);filtered[y*w+x]=value;}}
 d_recursiveGaussian_rgba<<<(w+127)/128,128>>>(input,output,w,h,a0,a1,a2,a3,b1,b2,cp,cn);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(actual.data(),output,(count+16)*4,cudaMemcpyDeviceToHost));for(int i=0;i<count+16;i++){if(i>=count&&actual[i]!=0xdeadbeef)return 1;for(int c=0;c<4;c++){int error=abs(int((actual[i]>>(8*c))&255)-int((filtered[i]>>(8*c))&255));maxError=std::max(maxError,error);if(error>2){printf("FAIL filter pixel=%d channel=%d error=%d\n",i,c,error);return 1;}}}
 for(int y=0;y<h;y++)for(int x=0;x<w;x++)expected[x*h+y]=filtered[y*w+x];d_transpose<<<dim3((w+15)/16,(h+15)/16),dim3(16,16)>>>(input,output,w,h);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(actual.data(),input,(count+16)*4,cudaMemcpyDeviceToHost));for(int i=0;i<count+16;i++){if(i>=count&&actual[i]!=0xdeadbeef)return 1;for(int c=0;c<4;c++)if(abs(int((actual[i]>>(8*c))&255)-int((expected[i]>>(8*c))&255))>2)return 1;}std::swap(w,h);
 }printf("%dx%d PASS original Gaussian/transpose four-pass pipeline, independent reference, guards; max channel error %d/255\n",w,h,maxError);CHECK(cudaFree(input));CHECK(cudaFree(output));
 }return 0;}

