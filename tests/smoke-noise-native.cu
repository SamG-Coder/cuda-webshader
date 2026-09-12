// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
cudaArray* noiseArray;cudaTextureObject_t noiseTex;
#include "smoke-noise-host.cuh"
#include "smoke-noise-kernel.cuh"
#include "smoke-noise-probe.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){
 srand(1);createNoiseTexture(64,64,64);
 std::vector<float4> data(64*64*64);cudaMemcpy3DParms read={};read.srcArray=noiseArray;read.dstPtr=make_cudaPitchedPtr(data.data(),64*sizeof(float4),64,64);read.extent=make_cudaExtent(64,64,64);read.kind=cudaMemcpyDeviceToHost;CHECK(cudaMemcpy3D(&read));
 FILE* file=fopen("reports/smoke-noise-input.bin","wb");if(!file)return 2;fwrite(data.data(),sizeof(float4),data.size(),file);fclose(file);
 std::vector<float4> coordinates;
 for(int z=0;z<8;z++)for(int y=0;y<8;y++)for(int x=0;x<8;x++)coordinates.push_back(make_float4((x*8+.5f)/64,(y*8+.5f)/64,(z*8+.5f)/64,0));
 for(int i=0;i<257;i++)coordinates.push_back(make_float4((i%17)*.139f-.7f,(i%13)*.217f-.9f,(i%11)*.329f-.8f,0));
 file=fopen("reports/smoke-noise-coordinates.bin","wb");if(!file)return 2;fwrite(coordinates.data(),sizeof(float4),coordinates.size(),file);fclose(file);
 float4 *input,*output;CHECK(cudaMalloc(&input,coordinates.size()*sizeof(float4)));CHECK(cudaMalloc(&output,coordinates.size()*sizeof(float4)));CHECK(cudaMemcpy(input,coordinates.data(),coordinates.size()*sizeof(float4),cudaMemcpyHostToDevice));std::vector<float4> result(coordinates.size());
 for(int mode=0;mode<4;mode++){
  CHECK(cudaDestroyTextureObject(noiseTex));cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=noiseArray;
  cudaTextureDesc settings={};settings.normalizedCoords=mode!=3;settings.filterMode=mode==1||mode==3?cudaFilterModePoint:cudaFilterModeLinear;settings.addressMode[0]=settings.addressMode[1]=settings.addressMode[2]=mode<2?cudaAddressModeWrap:cudaAddressModeClamp;settings.readMode=cudaReadModeElementType;CHECK(cudaCreateTextureObject(&noiseTex,&resource,&settings,nullptr));
  sampleSmokeNoise<<<(coordinates.size()+127)/128,128>>>(input,output,noiseTex,coordinates.size());CHECK(cudaGetLastError());CHECK(cudaMemcpy(result.data(),output,result.size()*sizeof(float4),cudaMemcpyDeviceToHost));
  char path[128];snprintf(path,sizeof(path),"reports/smoke-noise-%d-native.bin",mode);file=fopen(path,"wb");if(!file)return 2;fwrite(result.data(),sizeof(float4),result.size(),file);fclose(file);printf("Captured noise3D: mode=%d, coordinates=%zu, float4 volume=64³.\n",mode,result.size());
 }
 CHECK(cudaFree(input));CHECK(cudaFree(output));CHECK(cudaDestroyTextureObject(noiseTex));CHECK(cudaFreeArray(noiseArray));return 0;
}
