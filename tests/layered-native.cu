// MIT capture harness. Original NVIDIA kernel body is unchanged.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <helper_cuda.h>
#include "layered-kernel.cuh"
template<class T>void save(const char*path,const T*data,size_t n){FILE*f=fopen(path,"wb");if(!f||fwrite(data,sizeof(T),n,f)!=n)exit(1);fclose(f);}
int main(){
 const unsigned width=512,height=512,layers=5,N=width*height*layers;std::vector<float> input(N),output(N);for(unsigned i=0;i<N;i++)input[i]=float(i%(width*height));
 cudaArray_t array;auto format=cudaCreateChannelDesc<float>();checkCudaErrors(cudaMalloc3DArray(&array,&format,make_cudaExtent(width,height,layers),cudaArrayLayered));
 cudaMemcpy3DParms copy={};copy.srcPtr=make_cudaPitchedPtr(input.data(),width*sizeof(float),width,height);copy.dstArray=array;copy.extent=make_cudaExtent(width,height,layers);copy.kind=cudaMemcpyHostToDevice;checkCudaErrors(cudaMemcpy3D(&copy));
 cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;cudaTextureDesc desc={};desc.normalizedCoords=true;desc.filterMode=cudaFilterModeLinear;desc.addressMode[0]=desc.addressMode[1]=cudaAddressModeWrap;desc.readMode=cudaReadModeElementType;
 cudaTextureObject_t texture;checkCudaErrors(cudaCreateTextureObject(&texture,&resource,&desc,nullptr));float*result;checkCudaErrors(cudaMalloc(&result,N*sizeof(float)));
 for(unsigned layer=0;layer<layers;layer++)transformKernel<<<dim3(64,64,1),dim3(8,8,1)>>>(result,width,height,layer,texture);
 checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(output.data(),result,N*sizeof(float),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<N;i++)if(output[i]!=-input[i]+float(i/(width*height))){fprintf(stderr,"Layer mismatch at %u\n",i);return 1;}
 save("reports/layered-input.bin",input.data(),N);save("reports/layered-native.bin",output.data(),N);
 printf("width=%u height=%u layers=%u values=%u original_expected_transform=passed\n",width,height,layers,N);
 checkCudaErrors(cudaFree(result));checkCudaErrors(cudaDestroyTextureObject(texture));checkCudaErrors(cudaFreeArray(array));
}
