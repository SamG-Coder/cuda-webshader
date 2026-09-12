// SPDX-License-Identifier: MIT — host validation; kernels retain NVIDIA's notice.
#include <cuda_runtime.h>
#include <helper_math.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <string>
#include "box-filter-kernel.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(int scenario=0;scenario<4;scenario++){
 const int w=scenario<2?64:1024,h=scenario<2?37:1024,r=scenario==0?0:scenario==1?3:scenario==2?14:22;std::vector<uchar4> pixels(w*h);
 if(scenario<2){for(int i=0;i<w*h;i++)pixels[i]=make_uchar4((i*37)%256,(i*37+73)%256,(i*37+146)%256,255);}else{
  std::ifstream f("showcases/box-filter/teapot1024.ppm",std::ios::binary);std::string magic,line;int iw,ih,max;f>>magic;f>>std::ws;while(f.peek()=='#'){std::getline(f,line);f>>std::ws;}f>>iw>>ih>>max;f.get();if(magic!="P6"||iw!=w||ih!=h||max!=255)return 2;std::vector<unsigned char> rgb(w*h*3);f.read((char*)rgb.data(),rgb.size());if(f.gcount()!=rgb.size())return 2;for(int i=0;i<w*h;i++)pixels[i]=make_uchar4(rgb[i*3],rgb[i*3+1],rgb[i*3+2],255);
 }
 cudaArray_t array;auto channel=cudaCreateChannelDesc<uchar4>();CHECK(cudaMallocArray(&array,&channel,w,h));CHECK(cudaMemcpy2DToArray(array,0,0,pixels.data(),w*4,w*4,h,cudaMemcpyHostToDevice));cudaResourceDesc res={};res.resType=cudaResourceTypeArray;res.res.array.array=array;cudaTextureDesc desc={};desc.filterMode=cudaFilterModeLinear;desc.addressMode[0]=desc.addressMode[1]=cudaAddressModeClamp;desc.readMode=cudaReadModeNormalizedFloat;cudaTextureObject_t tex;CHECK(cudaCreateTextureObject(&tex,&res,&desc,nullptr));
 unsigned *row,*output;std::vector<unsigned> data(w*h+16,0xdeadbeef);CHECK(cudaMalloc(&row,data.size()*4));CHECK(cudaMalloc(&output,data.size()*4));CHECK(cudaMemcpy(row,data.data(),data.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(output,data.data(),data.size()*4,cudaMemcpyHostToDevice));
 d_boxfilter_rgba_x<<<(h+63)/64,64>>>(row,w,h,r,tex);CHECK(cudaGetLastError());d_boxfilter_rgba_y<<<w/64,64>>>(row,output,w,h,r);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 for(int stage=0;stage<2;stage++){CHECK(cudaMemcpy(data.data(),stage?output:row,data.size()*4,cudaMemcpyDeviceToHost));for(int i=w*h;i<w*h+16;i++)if(data[i]!=0xdeadbeef)return 3;char path[128];snprintf(path,sizeof(path),"reports/box-filter-native-%d-%d.bin",scenario,stage);FILE* f=fopen(path,"wb");if(!f||fwrite(data.data(),4,data.size(),f)!=data.size())return 2;fclose(f);}
 printf("PASS case=%d %dx%d radius=%d: original RGBA row/column kernels; both guards intact.\n",scenario,w,h,r);CHECK(cudaFree(row));CHECK(cudaFree(output));CHECK(cudaDestroyTextureObject(tex));CHECK(cudaFreeArray(array));
 }
 const int w=64,h=32,count=w*h;std::vector<float> input(count);for(int i=0;i<count;i++)input[i]=float((i*13)%256)/256.0f;float *src,*dst;CHECK(cudaMalloc(&src,count*4));CHECK(cudaMalloc(&dst,(count+16)*4));CHECK(cudaMemcpy(src,input.data(),count*4,cudaMemcpyHostToDevice));cudaArray_t array;auto desc=cudaCreateChannelDesc<float>();CHECK(cudaMallocArray(&array,&desc,w,h));CHECK(cudaMemcpy2DToArray(array,0,0,input.data(),w*4,w*4,h,cudaMemcpyHostToDevice));cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;cudaTextureDesc sampler={};sampler.filterMode=cudaFilterModeLinear;sampler.addressMode[0]=sampler.addressMode[1]=cudaAddressModeClamp;sampler.readMode=cudaReadModeElementType;cudaTextureObject_t texture;CHECK(cudaCreateTextureObject(&texture,&resource,&sampler,nullptr));
 for(int stage=0;stage<4;stage++){std::vector<float> output(count+16,-1234.5f);CHECK(cudaMemcpy(dst,output.data(),output.size()*4,cudaMemcpyHostToDevice));if(stage==0)d_boxfilter_x_global<<<1,32>>>(src,dst,w,h,3);if(stage==1)d_boxfilter_y_global<<<2,32>>>(src,dst,w,h,3);if(stage==2)d_boxfilter_x_tex<<<1,32>>>(dst,w,h,3,texture);if(stage==3)d_boxfilter_y_tex<<<2,32>>>(dst,w,h,3,texture);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output.data(),dst,output.size()*4,cudaMemcpyDeviceToHost));for(int i=count;i<count+16;i++)if(output[i]!=-1234.5f)return 3;char path[128];snprintf(path,sizeof(path),"reports/box-filter-float-native-%d.bin",stage);FILE* f=fopen(path,"wb");if(!f||fwrite(output.data(),4,output.size(),f)!=output.size())return 2;fclose(f);printf("PASS scalar stage=%d: 64x32 radius 3; guards intact.\n",stage);}
 CHECK(cudaDestroyTextureObject(texture));CHECK(cudaFreeArray(array));CHECK(cudaFree(src));CHECK(cudaFree(dst));return 0;}
