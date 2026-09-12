#include <cstdio>
#include <helper_cuda.h>
#include "../.local/raytracing-cuda/sphere.h"
#include "../.local/raytracing-cuda/hitable_list.h"
#include "pathtracer-list-kernels.cuh"
int main(){
 hitable **objects,**world;float *out,values[8];
 checkCudaErrors(cudaMalloc(&objects,2*sizeof(hitable*)));
 checkCudaErrors(cudaMalloc(&world,sizeof(hitable*)));
 checkCudaErrors(cudaMalloc(&out,4*sizeof(float)));
 create_list<<<1,1>>>(objects,world);checkCudaErrors(cudaDeviceSynchronize());
 trace_list<<<1,2>>>(world,out);checkCudaErrors(cudaDeviceSynchronize());
 checkCudaErrors(cudaMemcpy(values,out,16,cudaMemcpyDeviceToHost));
 update_list<<<1,1>>>(objects);checkCudaErrors(cudaDeviceSynchronize());
 trace_list<<<1,2>>>(world,out);checkCudaErrors(cudaDeviceSynchronize());
 checkCudaErrors(cudaMemcpy(values+4,out,16,cudaMemcpyDeviceToHost));
 free_list<<<1,1>>>(objects,world);checkCudaErrors(cudaDeviceSynchronize());
 FILE*f=fopen("reports/pathtracer-list-native.bin","wb");if(!f)return 2;
 fwrite(values,4,8,f);fclose(f);for(float v:values)printf("%g ",v);printf("\n");
 checkCudaErrors(cudaFree(objects));checkCudaErrors(cudaFree(world));checkCudaErrors(cudaFree(out));
}
