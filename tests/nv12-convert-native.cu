// MIT capture harness: unchanged conversion helper/kernel, full default batch.
#include <cstdio>
#include <cstring>
#include <vector>
#include <helper_cuda.h>
#include "nv12-convert-kernel.cuh"
int main(){
 const int width=1920,height=1080,batch=24,frameBytes=width*height*3/2,count=width*height*3;
 std::vector<unsigned char> frame(frameBytes);FILE*f=fopen("reports/nv12-input.bin","rb");if(!f||fread(frame.data(),1,frame.size(),f)!=frame.size())return 1;fclose(f);
 unsigned char* input;float* output;checkCudaErrors(cudaMalloc(&input,size_t(frameBytes)*batch));checkCudaErrors(cudaMalloc(&output,size_t(count)*batch*4));
 for(int i=0;i<batch;i++)checkCudaErrors(cudaMemcpy(input+size_t(i)*frameBytes,frame.data(),frameBytes,cudaMemcpyHostToDevice));
 nv12ToBGRplanarBatchKernel<<<dim3(8,54,24),dim3(64,10,1)>>>(input,width,output,width*4,width,height,batch);
 checkCudaErrors(cudaDeviceSynchronize());
 std::vector<float> first(count),current(count);
 for(int i=0;i<batch;i++){
  checkCudaErrors(cudaMemcpy(current.data(),output+size_t(i)*count,count*4,cudaMemcpyDeviceToHost));
  if(i==0)first=current;else if(memcmp(first.data(),current.data(),count*4)){fprintf(stderr,"Different frame %d\n",i);return 1;}
 }
 f=fopen("reports/nv12-convert-native.bin","wb");if(!f||fwrite(first.data(),4,count,f)!=count)return 1;fclose(f);
 printf("Original NV12 conversion: 1920x1080, 24 batches, 149299200 output values checked, all frames bitwise identical\n");
 checkCudaErrors(cudaFree(input));checkCudaErrors(cudaFree(output));
}
