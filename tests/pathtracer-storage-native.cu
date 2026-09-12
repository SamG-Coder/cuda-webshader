#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "../.local/raytracing-cuda/material.h"
#include "../.local/raytracing-cuda/sphere.h"
#include "pathtracer-storage-kernels.cuh"
int main(){curandState*states;vec3*fb;unsigned int*bits;hitable**world;
checkCudaErrors(cudaMalloc(&states,512*sizeof(curandState)));checkCudaErrors(cudaMalloc(&fb,512*sizeof(vec3)));checkCudaErrors(cudaMalloc(&bits,512*4));checkCudaErrors(cudaMalloc(&world,512*sizeof(hitable*)));
std::vector<float>v(512*3*3);std::vector<unsigned int>b(512*3);
render_init<<<dim3(4,2),dim3(8,8)>>>(32,16,states);checkCudaErrors(cudaDeviceSynchronize());
for(int cycle=0;cycle<3;cycle++){if(cycle==2){seed_wide<<<8,64>>>(states);checkCudaErrors(cudaDeviceSynchronize());}draw_states<<<8,64>>>(states,fb,bits);checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(v.data()+cycle*1536,fb,1536*4,cudaMemcpyDeviceToHost));checkCudaErrors(cudaMemcpy(b.data()+cycle*512,bits,512*4,cudaMemcpyDeviceToHost));}
create_storage_scene<<<8,64>>>(world);checkCudaErrors(cudaDeviceSynchronize());free_storage_scene<<<8,64>>>(world);checkCudaErrors(cudaDeviceSynchronize());std::vector<hitable*>p(512);checkCudaErrors(cudaMemcpy(p.data(),world,512*sizeof(hitable*),cudaMemcpyDeviceToHost));for(auto x:p)if(x)return 4;
FILE*f=fopen("reports/pathtracer-storage-native.bin","wb");if(!f)return 2;fwrite(v.data(),4,v.size(),f);fclose(f);f=fopen("reports/pathtracer-storage-native-bits.bin","wb");if(!f)return 3;fwrite(b.data(),4,b.size(),f);fclose(f);
checkCudaErrors(cudaFree(states));checkCudaErrors(cudaFree(fb));checkCudaErrors(cudaFree(bits));checkCudaErrors(cudaFree(world));printf("Original render_init; 3 RNG buffer passes; 512 sphere/material pairs freed through downcast\n");}
