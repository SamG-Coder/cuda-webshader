#include <cstdio>
#include <vector>
#include "../.local/nvidia-audit/cpp/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_kernel.cu"
extern "C" void initQuasirandomGenerator(unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]);
template<class T> void save(const char*path,const T*data,size_t n){FILE*f=fopen(path,"wb");if(!f||fwrite(data,sizeof(T),n,f)!=n){fprintf(stderr,"Capture write failed\n");exit(1);}fclose(f);}
int main(){unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION];initQuasirandomGenerator(table);initTableGPU(table);save("reports/quasirandom-table.bin",&table[0][0],93);
 const unsigned int counts[]={1048576,1025,1025},seeds[]={0,1,0xfffffff0u};
 for(int c=0;c<3;c++){unsigned int n=counts[c];float*out;checkCudaErrors(cudaMalloc(&out,n*3*sizeof(float)));quasirandomGeneratorGPU(out,seeds[c],n);checkCudaErrors(cudaDeviceSynchronize());std::vector<float> host(n*3);checkCudaErrors(cudaMemcpy(host.data(),out,host.size()*sizeof(float),cudaMemcpyDeviceToHost));char path[128];snprintf(path,sizeof(path),"reports/quasirandom-%d-native.bin",c);save(path,host.data(),host.size());checkCudaErrors(cudaFree(out));printf("case %d seed %u count %u components %zu captured\n",c,seeds[c],n,host.size());}
}
