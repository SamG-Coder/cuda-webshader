// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "fluids-pitched-kernels.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA error line %d: %s\n",__LINE__,cudaGetErrorString(e));return 2;}}while(0)
bool save(const char*path,const void*data,size_t bytes){FILE*f=fopen(path,"wb");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;fclose(f);return ok;}
bool load(const char*path,void*data,size_t bytes){FILE*f=fopen(path,"rb");if(!f)return false;bool ok=fread(data,1,bytes,f)==bytes;fclose(f);return ok;}
int main(){
 const int width=19,height=13,pdx=24,records=pdx*height+4,count=width*height;
 std::vector<float2> velocity(records,make_float2(-77,-77)),particles(count);std::vector<float> vx(pdx*height+16),vy(vx.size());
 for(int y=0;y<height;y++)for(int x=0;x<width;x++){velocity[y*pdx+x]=make_float2((x-9)*.01f,(y-6)*.02f);particles[y*width+x]=make_float2((x+.5f)/width,(y+.5f)/height);}
 if(!save("reports/fluids-pitched-input.bin",velocity.data(),records*8)||!save("reports/fluids-particles-input.bin",particles.data(),count*8))return 2;
 if(!load("reports/fluids-advect-1-0-native.bin",vx.data(),vx.size()*4)||!load("reports/fluids-advect-1-1-native.bin",vy.data(),vy.size()*4))return 2;
 float2 *v,*p;float *dx,*dy;CHECK(cudaMalloc(&v,records*8));CHECK(cudaMalloc(&p,count*8));CHECK(cudaMalloc(&dx,vx.size()*4));CHECK(cudaMalloc(&dy,vy.size()*4));
 CHECK(cudaMemcpy(v,velocity.data(),records*8,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(p,particles.data(),count*8,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,vx.data(),vx.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dy,vy.data(),vy.size()*4,cudaMemcpyHostToDevice));
 addForces_k<<<1,dim3(5,5)>>>(v,width,height,3,2,.5f,-.3f,2,pdx*8);CHECK(cudaMemcpy(velocity.data(),v,records*8,cudaMemcpyDeviceToHost));if(!save("reports/fluids-forces-native.bin",velocity.data(),records*8))return 2;
 updateVelocity_k<<<dim3(2,2),dim3(16,4)>>>(v,dx,dy,width,pdx,height,3,pdx*8);CHECK(cudaMemcpy(velocity.data(),v,records*8,cudaMemcpyDeviceToHost));if(!save("reports/fluids-update-native.bin",velocity.data(),records*8))return 2;
 for(int step=0;step<5;step++)advectParticles_k<<<dim3(2,2),dim3(16,4)>>>(p,v,width,height,500.f,3,pdx*8);
 CHECK(cudaMemcpy(particles.data(),p,count*8,cudaMemcpyDeviceToHost));if(!save("reports/fluids-particles-native.bin",particles.data(),count*8))return 2;
 CHECK(cudaFree(v));CHECK(cudaFree(p));CHECK(cudaFree(dx));CHECK(cudaFree(dy));puts("Captured original forces, velocity update and five particle-advection steps; 19x13 domain, 192-byte pitch.");return 0;
}
