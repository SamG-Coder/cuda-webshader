// MIT harness. The native run uses the complete unchanged upstream vec3.h.
#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include "../.local/raytracing-cuda/vec3.h"
__global__ void values(float*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 vec3 a(float(i)*0.25f,float(i%7)-3.0f,float(i%11)*0.5f),b=a;
 a.e[0]=-99.0f;
 out[i*17]=b.x();out[i*17+1]=b.y();out[i*17+2]=b.z();
 out[i*17+3]=b.r();out[i*17+4]=b.g();out[i*17+5]=b.b();out[i*17+6]=b.squared_length();
  vec3 positive=+b,negative=-b;out[i*17+7]=positive.x();out[i*17+8]=negative.x();out[i*17+9]=negative.y();out[i*17+10]=negative.z();out[i*17+11]=b[i%3];vec3 assigned;assigned=b;b.e[1]=-88.0f;out[i*17+12]=assigned.y();out[i*17+13]=b.y();assigned[i%3]=float(i)*0.5f;assigned[(i+1)%3]+=2.0f;out[i*17+14]=assigned.x();out[i*17+15]=assigned.y();out[i*17+16]=assigned.z();
}
int main(){float*out;std::vector<float> data(512*17);checkCudaErrors(cudaMalloc(&out,data.size()*4));values<<<8,64>>>(out);checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(data.data(),out,data.size()*4,cudaMemcpyDeviceToHost));FILE*f=fopen("reports/pathtracer-value-native.bin","wb");if(!f||fwrite(data.data(),4,data.size(),f)!=data.size())return 1;fclose(f);checkCudaErrors(cudaFree(out));printf("Original complete vec3 header: 512 constructor/copy/accessor/squared-length cases, 8704 values\n");}
