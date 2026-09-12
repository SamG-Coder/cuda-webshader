// MIT native capture harness; original NVIDIA implementations are included unchanged.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <helper_cuda.h>
#include "../.local/nvidia-audit/cpp/5_Domain_Specific/fastWalshTransform/fastWalshTransform_kernel.cuh"
extern "C" void dyadicConvolutionCPU(float*,float*,float*,int,int);
template<class T>void save(const char* path,const T* data,size_t n){FILE*f=fopen(path,"wb");if(!f||fwrite(data,sizeof(T),n,f)!=n){fprintf(stderr,"Capture write failed\n");exit(1);}fclose(f);}
int main(){
 const int log2N=23,N=1<<log2N,K=128;std::vector<float> input(N),kernel(K),output(N),cpu(N);
 srand(2007);for(float&v:kernel)v=float(rand())/float(RAND_MAX);for(float&v:input)v=float(rand())/float(RAND_MAX);
 float *data,*filter;checkCudaErrors(cudaMalloc(&data,N*4));checkCudaErrors(cudaMalloc(&filter,N*4));
 checkCudaErrors(cudaMemcpy(data,input.data(),N*4,cudaMemcpyHostToDevice));checkCudaErrors(cudaMemset(filter,0,N*4));checkCudaErrors(cudaMemcpy(filter,kernel.data(),K*4,cudaMemcpyHostToDevice));
 fwtBatchGPU(data,1,log2N);fwtBatchGPU(filter,1,log2N);modulateGPU(data,filter,N);fwtBatchGPU(data,1,log2N);
 checkCudaErrors(cudaDeviceSynchronize());checkCudaErrors(cudaMemcpy(output.data(),data,N*4,cudaMemcpyDeviceToHost));
 dyadicConvolutionCPU(cpu.data(),input.data(),kernel.data(),log2N,7);
 double delta=0,reference=0,maxError=0;for(int i=0;i<N;i++){if(!std::isfinite(output[i]))return 1;const double e=double(cpu[i])-output[i];delta+=e*e;reference+=double(cpu[i])*cpu[i];maxError=fmax(maxError,fabs(e));}
 const double l2=sqrt(delta/reference);if(l2>=1e-6){fprintf(stderr,"Original CPU comparison failed %.9g\n",l2);return 1;}
 save("reports/walsh-full-input.bin",input.data(),N);save("reports/walsh-full-kernel.bin",kernel.data(),K);save("reports/walsh-full-native.bin",output.data(),N);
 printf("values=%d kernel=%d native_vs_original_cpu_l2=%.9g max_absolute_error=%.9g passed\n",N,K,l2,maxError);
 checkCudaErrors(cudaFree(data));checkCudaErrors(cudaFree(filter));
}
