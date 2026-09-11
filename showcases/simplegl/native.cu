// MIT-licensed native verification host; kernel.cu retains NVIDIA's BSD license.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "kernel.cu"
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(){try{cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));printf("GPU: %s\n",prop.name);for(unsigned n:{128u,256u,512u,1024u}){float4* p;check(cudaMalloc((void**)&p,n*n*sizeof(float4)));std::vector<float4> host(n*n);for(float t:{0.0f,1.25f,7.5f}){simple_vbo_kernel<<<dim3(n/8,n/8),dim3(8,8)>>>(p,n,n,t);check(cudaGetLastError());check(cudaMemcpy(host.data(),p,n*n*sizeof(float4),cudaMemcpyDeviceToHost));double maxError=0;for(unsigned y=0;y<n;y++)for(unsigned x=0;x<n;x++){double u=double(x)/n*2-1,v=double(y)/n*2-1,w=std::sin(u*4+t)*std::cos(v*4+t)*0.5;const auto a=host[y*n+x];double error=std::fabs(a.y-w);if(!std::isfinite(a.y)||error>0.000003||a.x!=float(u)||a.z!=float(v)||a.w!=1)throw std::runtime_error("Output mismatch");maxError=std::fmax(maxError,error);}printf("PASS %ux%u time=%.2f max_error=%.9g\n",n,n,t,maxError);}check(cudaFree(p));}puts("12/12 native NVIDIA simpleGL kernel checks passed.");return 0;}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
