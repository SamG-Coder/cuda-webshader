// Original project test harness (MIT); included NVIDIA kernel retains BSD-3-Clause.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include "kernels/21.cu"
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){
 int failures=0;
 for(int n:{0,2,10,258,259,1024}){
  int count=((n+1)/2)*2+16;std::vector<float> s(count),x(count),t(count),call(count,-12345),put(count,-12345);float r=.02f,v=.3f;
  for(int i=0;i<count;i++){s[i]=5+(i*7%397);x[i]=1+(i*11%431);t[i]=.03125f+(i%64)/8.f;}
  float *ds,*dx,*dt,*dc,*dp;
  CHECK(cudaMalloc(&ds,count*4));CHECK(cudaMalloc(&dx,count*4));CHECK(cudaMalloc(&dt,count*4));CHECK(cudaMalloc(&dc,count*4));CHECK(cudaMalloc(&dp,count*4));
  CHECK(cudaMemcpy(ds,s.data(),count*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,x.data(),count*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dt,t.data(),count*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dc,call.data(),count*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dp,put.data(),count*4,cudaMemcpyHostToDevice));
  BlackScholesGPU<<<n?((n+255)/256):1,128>>>((float2*)dc,(float2*)dp,(float2*)ds,(float2*)dx,(float2*)dt,r,v,n);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(call.data(),dc,count*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(put.data(),dp,count*4,cudaMemcpyDeviceToHost));
  double maximum=0;int errors=0;
  for(int i=0;i<count;i++){
   double ec=-12345,ep=-12345;
   if(i<(n/2)*2){double d1=(log(double(s[i])/x[i])+(double(r)+double(v)*v/2)*t[i])/(v*sqrt(double(t[i]))),d2=d1-v*sqrt(double(t[i])),discount=x[i]*exp(-double(r)*t[i]);ec=s[i]*.5*erfc(-d1/sqrt(2.))-discount*.5*erfc(-d2/sqrt(2.));ep=discount*.5*erfc(d2/sqrt(2.))-s[i]*.5*erfc(d1/sqrt(2.));}
   double ce=fabs(call[i]-ec),pe=fabs(put[i]-ep);maximum=fmax(maximum,fmax(ce,pe));
   if(!std::isfinite(call[i])||!std::isfinite(put[i])||ce>.0002+.00002*fabs(ec)||pe>.0002+.00002*fabs(ep)||(i>=(n/2)*2&&(call[i]!=-12345||put[i]!=-12345)))errors++;
  }
  printf("options=%d both outputs + guards: %s maxError=%.9g\n",n,errors?"FAIL":"PASS",maximum);failures+=errors;
  CHECK(cudaFree(ds));CHECK(cudaFree(dx));CHECK(cudaFree(dt));CHECK(cudaFree(dc));CHECK(cudaFree(dp));
 }
 return failures?1:0;
}
