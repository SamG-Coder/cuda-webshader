// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <helper_math.h>
#include "float4-surface-kernel.cuh"
#include "volume-transfer-kernel.cuh"
#define CHECK(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;} } while(0)
int main() {
    constexpr int width=64;
    std::vector<float4> input(width),output(width);
    for(int i=0;i<width;i++) input[i]=make_float4(i*.125f,-i*.25f,(i%7)*.0625f,1.f-i*.03125f);
    float4 *deviceInput,*deviceOutput;
    CHECK(cudaMalloc(&deviceInput,width*sizeof(float4)));
    CHECK(cudaMalloc(&deviceOutput,width*sizeof(float4)));
    CHECK(cudaMemcpy(deviceInput,input.data(),width*sizeof(float4),cudaMemcpyHostToDevice));
    cudaArray_t array;auto desc=cudaCreateChannelDesc<float4>();
    CHECK(cudaMallocArray(&array,&desc,width,0,cudaArraySurfaceLoadStore));
    cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
    cudaSurfaceObject_t surface;CHECK(cudaCreateSurfaceObject(&surface,&resource));
    cudaTextureDesc settings={};settings.normalizedCoords=1;settings.filterMode=cudaFilterModeLinear;
    settings.addressMode[0]=cudaAddressModeClamp;settings.readMode=cudaReadModeElementType;
    cudaTextureObject_t texture;CHECK(cudaCreateTextureObject(&texture,&resource,&settings,nullptr));
    writeTransfer<<<2,32>>>(deviceInput,surface,width);CHECK(cudaGetLastError());
    readTransfer<<<2,32>>>(deviceOutput,texture,width);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output.data(),deviceOutput,width*sizeof(float4),cudaMemcpyDeviceToHost));
    FILE* file=fopen("reports/float4-surface-native.bin","wb");if(!file)return 2;
    fwrite(output.data(),sizeof(float4),width,file);fclose(file);
    for(int i=0;i<width;i++) for(int c=0;c<4;c++) if(((float*)&output[i])[c]!=((float*)&input[i])[c])return 3;
    readTransferZero<<<1,1>>>(deviceOutput,texture);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output.data(),deviceOutput,sizeof(float4),cudaMemcpyDeviceToHost));
    for(int c=0;c<4;c++)if(((float*)output.data())[c]!=((float*)input.data())[c])return 4;
    printf("PASS 64 float4 surface writes and texture reads; integer-zero sample matches first texel.\n");
    cudaArray_t integratedArray;CHECK(cudaMallocArray(&integratedArray,&desc,width,0,cudaArraySurfaceLoadStore));
    resource.res.array.array=integratedArray;
    cudaSurfaceObject_t integratedSurface;CHECK(cudaCreateSurfaceObject(&integratedSurface,&resource));
    d_integrate_trapezoidal<<<2,32>>>(make_cudaExtent(width,1,1),texture,integratedSurface);
    CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpyFromArray(output.data(),integratedArray,0,0,width*sizeof(float4),cudaMemcpyDeviceToHost));
    for(const auto& value:output)for(int c=0;c<4;c++)if(!std::isfinite(((const float*)&value)[c]))return 5;
    file=fopen("reports/volume-transfer-native.bin","wb");if(!file)return 2;
    fwrite(output.data(),sizeof(float4),width,file);fclose(file);
    printf("CAPTURED original d_integrate_trapezoidal: 64 finite float4 records.\n");
    CHECK(cudaDestroySurfaceObject(integratedSurface));CHECK(cudaFreeArray(integratedArray));
    for(int size:{37,1024}){
        CHECK(cudaMallocArray(&integratedArray,&desc,size,0,cudaArraySurfaceLoadStore));resource.res.array.array=integratedArray;CHECK(cudaCreateSurfaceObject(&integratedSurface,&resource));
        d_integrate_trapezoidal<<<(size+31)/32,32>>>(make_cudaExtent(size,0,0),texture,integratedSurface);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
        output.resize(size);CHECK(cudaMemcpyFromArray(output.data(),integratedArray,0,0,size*sizeof(float4),cudaMemcpyDeviceToHost));
        char path[128];snprintf(path,sizeof(path),"reports/volume-transfer-%d-native.bin",size);file=fopen(path,"wb");if(!file)return 2;fwrite(output.data(),sizeof(float4),size,file);fclose(file);
        printf("CAPTURED unchanged integration kernel: %d records.\n",size);CHECK(cudaDestroySurfaceObject(integratedSurface));CHECK(cudaFreeArray(integratedArray));
    }
    CHECK(cudaDestroyTextureObject(texture));CHECK(cudaDestroySurfaceObject(surface));CHECK(cudaFreeArray(array));
    CHECK(cudaFree(deviceInput));CHECK(cudaFree(deviceOutput));return 0;
}
