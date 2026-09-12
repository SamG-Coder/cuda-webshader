// MIT single-frame stage fixture; original NVIDIA wrapper/kernel are linked.
#include <cstdio>
#include <vector>
#include <helper_cuda.h>
extern "C" void resizeNV12Batch(unsigned char*,int,int,int,unsigned char*,int,int,int,int,cudaStream_t);
int main(){
 std::vector<unsigned char> input(1920*1080*3/2),output(640*480*3/2);
 FILE*f=fopen("reports/nv12-input.bin","rb");if(!f||fread(input.data(),1,input.size(),f)!=input.size())return 1;fclose(f);
 unsigned char *src,*dst;checkCudaErrors(cudaMalloc(&src,input.size()));checkCudaErrors(cudaMalloc(&dst,output.size()));
 checkCudaErrors(cudaMemcpy(src,input.data(),input.size(),cudaMemcpyHostToDevice));checkCudaErrors(cudaMemset(dst,0xa5,output.size()));
 resizeNV12Batch(src,1920,1920,1080,dst,640,640,480,1,0);checkCudaErrors(cudaDeviceSynchronize());
 checkCudaErrors(cudaMemcpy(output.data(),dst,output.size(),cudaMemcpyDeviceToHost));
 f=fopen("reports/nv12-resize-stage-native.bin","wb");if(!f||fwrite(output.data(),1,output.size(),f)!=output.size())return 1;fclose(f);
 printf("Original NV12 resize stage: one 1920x1080 frame -> 640x480, 460800 bytes captured; full batch defect not bypassed or fixed\n");
 checkCudaErrors(cudaFree(src));checkCudaErrors(cudaFree(dst));
}
