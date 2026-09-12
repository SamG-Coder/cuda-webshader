// MIT harness. The native run uses the complete unchanged upstream vec3.h.
#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include "../.local/raytracing-cuda/vec3.h"
__global__ void values(float*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 vec3 a(float(i)*0.25f,float(i%7)-3.0f,float(i%11)*0.5f),b=a;
 a.e[0]=-99.0f;
 out[i*38]=b.x();out[i*38+1]=b.y();out[i*38+2]=b.z();
 out[i*38+3]=b.r();out[i*38+4]=b.g();out[i*38+5]=b.b();out[i*38+6]=b.squared_length();
  vec3 positive=+b,negative=-b;out[i*38+7]=positive.x();out[i*38+8]=negative.x();out[i*38+9]=negative.y();out[i*38+10]=negative.z();out[i*38+11]=b[i%3];vec3 assigned;assigned=b;b.e[1]=-88.0f;out[i*38+12]=assigned.y();out[i*38+13]=b.y();assigned[i%3]=float(i)*0.5f;assigned[(i+1)%3]+=2.0f;out[i*38+14]=assigned.x();out[i*38+15]=assigned.y();out[i*38+16]=assigned.z();vec3 rhs(2.0f,4.0f,8.0f);vec3 step(8.0f,16.0f,32.0f);step+=rhs;out[i*38+17]=step.x();out[i*38+18]=step.y();out[i*38+19]=step.z();step-=rhs;out[i*38+20]=step.x();out[i*38+21]=step.y();out[i*38+22]=step.z();step*=rhs;out[i*38+23]=step.x();out[i*38+24]=step.y();out[i*38+25]=step.z();step/=rhs;out[i*38+26]=step.x();out[i*38+27]=step.y();out[i*38+28]=step.z();step*=2.0f;out[i*38+29]=step.x();out[i*38+30]=step.y();out[i*38+31]=step.z();step/=2.0f;out[i*38+32]=step.x();out[i*38+33]=step.y();out[i*38+34]=step.z();vec3 unit(3.0f,0.0f,4.0f);unit.make_unit_vector();out[i*38+35]=unit.x();out[i*38+36]=unit.y();out[i*38+37]=unit.z();
}
int main(){float*out;std::vector<float> data(512*38);checkCudaErrors(cudaMalloc(&out,data.size()*4));values<<<8,64>>>(out);checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(data.data(),out,data.size()*4,cudaMemcpyDeviceToHost));FILE*f=fopen("reports/pathtracer-value-native.bin","wb");if(!f||fwrite(data.data(),4,data.size(),f)!=data.size())return 1;fclose(f);checkCudaErrors(cudaFree(out));printf("Original complete vec3 header: 512 constructor/copy/accessor/squared-length cases, 19456 values\n");}
