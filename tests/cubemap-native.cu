// MIT capture harness; the NVIDIA kernel body is retained verbatim.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <helper_cuda.h>
#include "cubemap-kernel.cuh"
template<class T>void save(const char*path,const T*data,size_t n){FILE*f=fopen(path,"wb");if(!f||fwrite(data,sizeof(T),n,f)!=n)exit(1);fclose(f);}
__global__ void probeDirections(float* out,const float4* directions,cudaTextureObject_t tex){int i=threadIdx.x;float4 d=directions[i];out[i]=texCubemap<float>(tex,d.x,d.y,d.z);}
int main(){
 const unsigned width=64,N=width*width*6;std::vector<float> input(N),output(N);for(unsigned i=0;i<N;i++)input[i]=float(i);
 cudaArray_t array;auto format=cudaCreateChannelDesc<float>();checkCudaErrors(cudaMalloc3DArray(&array,&format,make_cudaExtent(width,width,6),cudaArrayCubemap));
 cudaMemcpy3DParms copy={};copy.srcPtr=make_cudaPitchedPtr(input.data(),width*sizeof(float),width,width);copy.dstArray=array;copy.extent=make_cudaExtent(width,width,6);copy.kind=cudaMemcpyHostToDevice;checkCudaErrors(cudaMemcpy3D(&copy));
 cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=array;
 cudaTextureDesc desc={};desc.normalizedCoords=true;desc.filterMode=cudaFilterModeLinear;desc.addressMode[0]=desc.addressMode[1]=desc.addressMode[2]=cudaAddressModeWrap;desc.readMode=cudaReadModeElementType;
 cudaTextureObject_t texture;checkCudaErrors(cudaCreateTextureObject(&texture,&resource,&desc,nullptr));
 float*result;checkCudaErrors(cudaMalloc(&result,N*sizeof(float)));transformKernel<<<dim3(8,8,1),dim3(8,8,1)>>>(result,width,texture);checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(output.data(),result,N*sizeof(float),cudaMemcpyDeviceToHost));
 for(unsigned i=0;i<N;i++)if(output[i]!=-input[i]){fprintf(stderr,"Face/orientation mismatch at %u\n",i);return 1;}
 save("reports/cubemap-input.bin",input.data(),N);save("reports/cubemap-native.bin",output.data(),N);
 printf("width=%u faces=6 values=%u original_expected_negation=passed\n",width,N);
 const float4 directions[]={{1,.999f,0,0},{1,-.999f,0,0},{-1,.999f,0,0},{-1,-.999f,0,0},{.999f,1,0,0},{-.999f,1,0,0},{.999f,-1,0,0},{-.999f,-1,0,0},{.999f,0,1,0},{-.999f,0,1,0},{.999f,0,-1,0},{-.999f,0,-1,0}};
 float4*coords;checkCudaErrors(cudaMalloc(&coords,sizeof(directions)));checkCudaErrors(cudaMemcpy(coords,directions,sizeof(directions),cudaMemcpyHostToDevice));
 float modes[2][12];for(int mode=0;mode<2;mode++){desc.seamlessCubemap=mode;cudaTextureObject_t probe;checkCudaErrors(cudaCreateTextureObject(&probe,&resource,&desc,nullptr));probeDirections<<<1,12>>>(result,coords,probe);checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(modes[mode],result,sizeof(modes[mode]),cudaMemcpyDeviceToHost));checkCudaErrors(cudaDestroyTextureObject(probe));}
 save("reports/cubemap-directions.bin",directions,12);save("reports/cubemap-edges-default.bin",modes[0],12);save("reports/cubemap-edges-seamless.bin",modes[1],12);
 int changed=0;for(int i=0;i<12;i++)changed+=modes[0][i]!=modes[1][i];printf("edge_probes=12 seamless_changes=%d\n",changed);checkCudaErrors(cudaFree(coords));
 checkCudaErrors(cudaFree(result));checkCudaErrors(cudaDestroyTextureObject(texture));checkCudaErrors(cudaFreeArray(array));
}
