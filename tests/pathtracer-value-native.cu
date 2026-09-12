// MIT harness. The native run uses the complete unchanged upstream vec3.h.
#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include "../.local/raytracing-cuda/vec3.h"
__global__ void values(float*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 vec3 a(float(i)*0.25f,float(i%7)-3.0f,float(i%11)*0.5f),b=a;
 a.e[0]=-99.0f;
 out[i*14]=b.x();out[i*14+1]=b.y();out[i*14+2]=b.z();
 out[i*14+3]=b.r();out[i*14+4]=b.g();out[i*14+5]=b.b();out[i*14+6]=b.squared_length();
  vec3 positive=+b,negative=-b;out[i*14+7]=positive.x();out[i*14+8]=negative.x();out[i*14+9]=negative.y();out[i*14+10]=negative.z();out[i*14+11]=b[i%3];vec3 assigned;assigned=b;b.e[1]=-88.0f;out[i*14+12]=assigned.y();out[i*14+13]=b.y();
}
int main(){float*out;std::vector<float> data(512*14);checkCudaErrors(cudaMalloc(&out,data.size()*4));values<<<8,64>>>(out);checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(data.data(),out,data.size()*4,cudaMemcpyDeviceToHost));FILE*f=fopen("reports/pathtracer-value-native.bin","wb");if(!f||fwrite(data.data(),4,data.size(),f)!=data.size())return 1;fclose(f);checkCudaErrors(cudaFree(out));printf("Original complete vec3 header: 512 constructor/copy/accessor/squared-length cases, 7168 values\n");}
