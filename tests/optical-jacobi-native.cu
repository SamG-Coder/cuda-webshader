#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include "optical-flow-kernels.cuh"
#define CHECK(x) do{if((x)!=cudaSuccess)return 2;}while(0)
int save(int c,const char* name,int step,const std::vector<float>&v){char path[128];sprintf(path,"reports/optical-jacobi-%d-%d-%s.bin",c,step,name);FILE*f=fopen(path,"wb");if(!f)return 2;fwrite(v.data(),4,v.size(),f);fclose(f);return 0;}
int main(){for(int c=0;c<2;c++){int w=c?128:67,h=c?96:19,s=c?128:96,n=s*h;std::vector<float> input[5],output(n);float *p[7];for(int a=0;a<7;a++){CHECK(cudaMalloc(&p[a],n*4));CHECK(cudaMemset(p[a],0,n*4));if(a<5){input[a].resize(n);for(int y=0;y<h;y++)for(int x=0;x<s;x++)input[a][y*s+x]=x<w?float(((x*13+y*7+a*11)%61)-30)/float(a<2?64:128):0;CHECK(cudaMemcpy(p[a],input[a].data(),n*4,cudaMemcpyHostToDevice));char name[16];sprintf(name,"input%d",a);if(save(c,name,0,input[a]))return 2;}}
for(int step=1;step<=500;step++){if(c)JacobiIteration<16,4><<<dim3((w+15)/16,(h+3)/4),dim3(16,4)>>>(p[0],p[1],p[2],p[3],p[4],w,h,s,.2f,p[5],p[6]);else JacobiIteration<32,6><<<dim3((w+31)/32,(h+5)/6),dim3(32,6)>>>(p[0],p[1],p[2],p[3],p[4],w,h,s,.2f,p[5],p[6]);std::swap(p[0],p[5]);std::swap(p[1],p[6]);if(step==1||step==8||step==500){for(int a=0;a<2;a++){CHECK(cudaMemcpy(output.data(),p[a],n*4,cudaMemcpyDeviceToHost));if(save(c,a?"v":"u",step,output))return 2;}printf("Captured optical Jacobi %dx%d, tile %dx%d, step %d.\n",w,h,c?16:32,c?4:6,step);}}
for(auto a:p)CHECK(cudaFree(a));}return 0;}
