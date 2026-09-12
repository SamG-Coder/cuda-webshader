// MIT harness. The native run uses the complete unchanged upstream vec3.h.
#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include "../.local/raytracing-cuda/vec3.h"
__global__ void values(float*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 vec3 a(float(i)*0.25f,float(i%7)-3.0f,float(i%11)*0.5f),b=a;
 a.e[0]=-99.0f;
 out[i*7]=b.x();out[i*7+1]=b.y();out[i*7+2]=b.z();
 out[i*7+3]=b.r();out[i*7+4]=b.g();out[i*7+5]=b.b();out[i*7+6]=b.squared_length();
}
int main(){float*out;std::vector<float> data(512*7);checkCudaErrors(cudaMalloc(&out,data.size()*4));values<<<8,64>>>(out);checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(data.data(),out,data.size()*4,cudaMemcpyDeviceToHost));FILE*f=fopen("reports/pathtracer-value-native.bin","wb");if(!f||fwrite(data.data(),4,data.size(),f)!=data.size())return 1;fclose(f);checkCudaErrors(cudaFree(out));printf("Original complete vec3 header: 512 constructor/copy/accessor/squared-length cases, 3584 values\n");}
