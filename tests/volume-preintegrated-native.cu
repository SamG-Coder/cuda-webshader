// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <helper_math.h>
#include <cstdio>
#include <vector>
#include "volume-preintegrated-render-kernel.cuh"
#include "volume-transfer-colors.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}}while(0)
struct Texture {cudaArray_t array;cudaTextureObject_t texture;cudaSurfaceObject_t surface;};
int makeTexture(Texture& t,int width,int height,int layers,bool layered){
 auto channel=cudaCreateChannelDesc<float4>();
 if(layered){CHECK(cudaMalloc3DArray(&t.array,&channel,make_cudaExtent(width,height,layers),cudaArrayLayered|cudaArraySurfaceLoadStore));}
 else{CHECK(cudaMallocArray(&t.array,&channel,width,height,cudaArraySurfaceLoadStore));}
 cudaResourceDesc r={};r.resType=cudaResourceTypeArray;r.res.array.array=t.array;
 CHECK(cudaCreateSurfaceObject(&t.surface,&r));cudaTextureDesc d={};d.normalizedCoords=1;d.filterMode=cudaFilterModeLinear;d.readMode=cudaReadModeElementType;d.addressMode[0]=d.addressMode[1]=d.addressMode[2]=cudaAddressModeClamp;
 CHECK(cudaCreateTextureObject(&t.texture,&r,&d,nullptr));return 0;
}
void destroy(Texture& t){cudaDestroyTextureObject(t.texture);cudaDestroySurfaceObject(t.surface);cudaFreeArray(t.array);}
int main(){
 Texture colors[2],integrated;for(int l=0;l<2;l++){if(makeTexture(colors[l],9,0,1,false))return 2;CHECK(cudaMemcpyToArray(colors[l].array,0,0,l?transferFunc1:transferFunc0,9*sizeof(float4),cudaMemcpyHostToDevice));}
 if(makeTexture(integrated,1024,0,1,false))return 2;
 std::vector<unsigned char> bucky(32*32*32);FILE* file=fopen("showcases/volume-render/Bucky.raw","rb");if(!file||fread(bucky.data(),1,bucky.size(),file)!=bucky.size())return 2;fclose(file);
 cudaArray_t volumeArray;auto bytes=cudaCreateChannelDesc<unsigned char>();CHECK(cudaMalloc3DArray(&volumeArray,&bytes,make_cudaExtent(32,32,32)));
 cudaMemcpy3DParms upload={};upload.srcPtr=make_cudaPitchedPtr(bucky.data(),32,32,32);upload.dstArray=volumeArray;upload.extent=make_cudaExtent(32,32,32);upload.kind=cudaMemcpyHostToDevice;CHECK(cudaMemcpy3D(&upload));
 cudaResourceDesc vr={};vr.resType=cudaResourceTypeArray;vr.res.array.array=volumeArray;cudaTextureDesc vd={};vd.normalizedCoords=1;vd.filterMode=cudaFilterModeLinear;vd.readMode=cudaReadModeNormalizedFloat;vd.addressMode[0]=vd.addressMode[1]=vd.addressMode[2]=cudaAddressModeClamp;cudaTextureObject_t volume;CHECK(cudaCreateTextureObject(&volume,&vr,&vd,nullptr));
 unsigned int* output;CHECK(cudaMalloc(&output,256*256*4));std::vector<unsigned int> image(256*256);
 for(int size:{32,1024}){
  Texture tables;if(makeTexture(tables,size,size,2,true))return 2;
  for(int layer=0;layer<2;layer++){
   d_integrate_trapezoidal<<<32,32>>>(make_cudaExtent(1024,0,0),colors[layer].texture,integrated.surface);CHECK(cudaGetLastError());
   d_preintegrate<<<dim3(size/8,size/8),dim3(8,8)>>>(layer,1024.f,make_cudaExtent(size,size,2),colors[layer].texture,integrated.texture,tables.surface);CHECK(cudaGetLastError());
  }
  CHECK(cudaDeviceSynchronize());
  if(size==32){std::vector<float4> data(size*size*2);cudaMemcpy3DParms read={};read.srcArray=tables.array;read.dstPtr=make_cudaPitchedPtr(data.data(),size*sizeof(float4),size,size);read.extent=make_cudaExtent(size,size,2);read.kind=cudaMemcpyDeviceToHost;CHECK(cudaMemcpy3D(&read));file=fopen("reports/volume-preintegrated-table-native.bin","wb");if(!file)return 2;fwrite(data.data(),sizeof(float4),data.size(),file);fclose(file);}
  for(int scenario=0;scenario<3;scenario++){
   float matrix[12]={1,0,0,0,0,1,0,0,0,0,1,4};
   if(scenario==1){float rotated[12]={.8660254f,0,.5f,2,0,1,0,0,-.5f,0,.8660254f,3.4641016f};memcpy(matrix,rotated,sizeof(matrix));}
   CHECK(cudaMemcpyToSymbol(c_invViewMatrix,matrix,sizeof(matrix)));CHECK(cudaMemset(output,0,256*256*4));
   if(scenario==2)d_render_preint_off<<<dim3(32,32),dim3(8,8)>>>(output,256,256,.05f,1,0,1,volume,colors[0].texture,tables.texture);
   else d_render_preint<<<dim3(32,32),dim3(8,8)>>>(output,256,256,.05f,1,0,1,volume,colors[0].texture,tables.texture);
   CHECK(cudaGetLastError());CHECK(cudaMemcpy(image.data(),output,256*256*4,cudaMemcpyDeviceToHost));char path[128];snprintf(path,sizeof(path),"reports/volume-preintegrated-%d-%d-native.bin",size,scenario);file=fopen(path,"wb");if(!file)return 2;fwrite(image.data(),4,image.size(),file);fclose(file);printf("Captured original renderer: transfer=%d²x2, scenario=%d, image=256².\n",size,scenario);
  }
  destroy(tables);
 }
 CHECK(cudaFree(output));CHECK(cudaDestroyTextureObject(volume));CHECK(cudaFreeArray(volumeArray));destroy(integrated);for(auto& t:colors)destroy(t);return 0;
}
