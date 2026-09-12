// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <helper_image.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include "sobel-shared-kernel.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA ERROR %s\n",cudaGetErrorString(e));return 2;}}while(0)
int main(){for(int scenario=0;scenario<4;scenario++){
 unsigned w=scenario==0?32:64,h=scenario==0?17:65;std::vector<unsigned char> data;
 if(scenario>=2){unsigned char* image=nullptr;if(!sdkLoadPGM("showcases/sobel/teapot.pgm",&image,&w,&h))return 2;data.assign(image,image+w*h);free(image);}else{data.resize(w*h);for(unsigned i=0;i<data.size();i++)data[i]=(i*37+(i/w)*13)&255;}
 unsigned pitch=w+(scenario<2?4:0),size=((pitch*h+19)/4)*4;float scale=scenario==3?.25f:1.f;
 cudaArray_t array;auto channel=cudaCreateChannelDesc<unsigned char>();CHECK(cudaMallocArray(&array,&channel,w,h));CHECK(cudaMemcpy2DToArray(array,0,0,data.data(),w,w,h,cudaMemcpyHostToDevice));cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;cudaTextureDesc desc={};desc.normalizedCoords=0;desc.filterMode=cudaFilterModePoint;desc.addressMode[0]=cudaAddressModeWrap;desc.readMode=cudaReadModeElementType;cudaTextureObject_t tex;CHECK(cudaCreateTextureObject(&tex,&resource,&desc,nullptr));unsigned char* dst;CHECK(cudaMalloc(&dst,size));
 for(int pass=0;pass<1;pass++){std::vector<unsigned char> out(size,165);CHECK(cudaMemcpy(dst,out.data(),size,cudaMemcpyHostToDevice));SobelShared<<<dim3((w+319)/320,(h+3)/4),dim3(16,4),2304>>>((uchar4*)dst,pitch,80,384,w,h,scale,tex);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(out.data(),dst,size,cudaMemcpyDeviceToHost));
 auto at=[&](int x,int y){x=x<0?0:x>=int(w)?w-1:x;y=y<0?0:y>=int(h)?h-1:y;return int(data[y*w+x]);};
 for(unsigned i=0;i<size;i++){int expected=165;int x=i%pitch,y=i/pitch;if(x<int(w)&&y<int(h)){if(pass==1)expected=int(scale*at(x,y));else{int horz=at(x+1,y-1)+2*at(x+1,y)+at(x+1,y+1)-at(x-1,y-1)-2*at(x-1,y)-at(x-1,y+1),vert=at(x-1,y-1)+2*at(x,y-1)+at(x+1,y-1)-at(x-1,y+1)-2*at(x,y+1)-at(x+1,y+1);expected=int(scale*(abs(horz)+abs(vert)));}expected=expected<0?0:expected>255?255:expected;}if(out[i]!=expected){printf("FAIL %d %d byte=%u got=%u expected=%d\n",scenario,pass,i,out[i],expected);return 3;}}
 char path[128];snprintf(path,sizeof(path),"reports/sobel-shared-native-%d-%d.bin",scenario,pass);FILE*f=fopen(path,"wb");if(!f||fwrite(out.data(),1,size,f)!=size)return 2;fclose(f);printf("PASS %s %ux%u pitch=%u scale=%g, all pixels and guards.\n","SobelShared",w,h,pitch,scale);
 }CHECK(cudaFree(dst));CHECK(cudaDestroyTextureObject(tex));CHECK(cudaFreeArray(array));}}
