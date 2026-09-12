// MIT harness for original BGR resize kernel: batch loop, filtering and padding.
#include <cstdio>
#include <vector>
#include <helper_cuda.h>
#include "nv12-bgr-kernel.cuh"
int main(){
 const int width=32,height=8,batch=24,dstPitch=20,dstHeight=4;
 std::vector<float> input(width*height*3*batch),output(dstPitch*dstHeight*3*batch,-99.0f);
 for(size_t i=0;i<input.size();i++)input[i]=float(i%width+(i/width)*64);
 cudaArray_t array;auto channel=cudaCreateChannelDesc<float>();
 checkCudaErrors(cudaMallocArray(&array,&channel,width,height*3*batch));
 checkCudaErrors(cudaMemcpy2DToArray(array,0,0,input.data(),width*4,width*4,height*3*batch,cudaMemcpyHostToDevice));
 cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
 cudaTextureDesc desc={};desc.filterMode=cudaFilterModeLinear;desc.readMode=cudaReadModeElementType;
 desc.addressMode[0]=desc.addressMode[1]=cudaAddressModeClamp;
 cudaTextureObject_t texture;checkCudaErrors(cudaCreateTextureObject(&texture,&resource,&desc,nullptr));
 float* out;checkCudaErrors(cudaMalloc(&out,output.size()*4));checkCudaErrors(cudaMemcpy(out,output.data(),output.size()*4,cudaMemcpyHostToDevice));
 resizeBGRplanarBatchKernel<<<dim3(1,1,4),dim3(32,32,1)>>>(texture,out,dstPitch,dstHeight,height,batch,2,2,0,0,width,height);
 checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(output.data(),out,output.size()*4,cudaMemcpyDeviceToHost));
 FILE*f=fopen("reports/nv12-bgr-stage-native.bin","wb");if(!f||fwrite(output.data(),4,output.size(),f)!=output.size())return 1;fclose(f);
 printf("Original BGR resize: 24 batches, 3 channels, 32x8 -> 16x4, pitch20, gridZ4, 5760 captured values\n");
 checkCudaErrors(cudaFree(out));checkCudaErrors(cudaDestroyTextureObject(texture));checkCudaErrors(cudaFreeArray(array));
}
