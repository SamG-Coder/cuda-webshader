// Real CUDA execution: the original kernel files are compiled by NVCC.
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <string>
#include "../kernels/saxpy.cu"
#include "../kernels/saxpy_vec4.cu"
#include "../kernels/matmul_naive.cu"
#include "../kernels/matmul_tiled.cu"
#include "../kernels/matmul_register.cu"
#include "../kernels/reduce_sum.cu"
#include "../kernels/convolution.cu"
#include "../kernels/histogram.cu"
#undef TILE
#include "../kernels/transpose.cu"
#include "../kernels/particles.cu"
__global__ void clear_bins(unsigned int* bins) { bins[threadIdx.x]=0u; }
inline void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
 void* p=nullptr;std::vector<char> host;
 explicit Buffer(size_t bytes):host(bytes){ck(cudaMalloc(&p,bytes));}
 explicit Buffer(const std::string& file){std::ifstream f(file,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Missing "+file);host.resize(size_t(f.tellg()));f.seekg(0);f.read(host.data(),host.size());ck(cudaMalloc(&p,host.size()));reset();}
 ~Buffer(){cudaFree(p);}
 Buffer(const Buffer&)=delete;
 void reset(){ck(cudaMemcpy(p,host.data(),host.size(),cudaMemcpyHostToDevice));}
 void verify(const std::string& file,bool integer=false){std::ifstream f(file,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Missing expected "+file);size_t size=size_t(f.tellg());f.seekg(0);std::vector<char> expected(size),actual(size);f.read(expected.data(),size);ck(cudaMemcpy(actual.data(),p,size,cudaMemcpyDeviceToHost));double maxError=0;
  for(size_t i=0;i<size/4;i++){double a=integer?reinterpret_cast<unsigned*>(actual.data())[i]:reinterpret_cast<float*>(actual.data())[i],e=integer?reinterpret_cast<unsigned*>(expected.data())[i]:reinterpret_cast<float*>(expected.data())[i];double error=std::abs(a-e);maxError=std::max(maxError,error);if(!std::isfinite(a)||error>(integer?0:0.001+0.001*std::abs(e)))throw std::runtime_error(file+" mismatch at "+std::to_string(i));}
  std::cerr<<"PASS "<<file<<" max_abs_error="<<maxError<<'\n';
 }
};
inline double med(std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];}
template<class Launch,class Reset> void benchmark(const char* key,int iterations,Launch launch,Reset reset){
 cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
 reset();for(int i=0;i<5;i++)launch(stream);ck(cudaStreamSynchronize(stream));
 // A graph queues the whole batch, analogous to a WebGPU command buffer, avoiding host launch starvation.
 cudaGraph_t graph;cudaGraphExec_t executable;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<iterations;i++)launch(stream);ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,0));
 ck(cudaGraphUpload(executable,stream));ck(cudaStreamSynchronize(stream));
 std::vector<double> gpu,wall,directGpu,directWall;
 for(int s=0;s<9;s++){
  reset();ck(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();ck(cudaEventRecord(start,stream));ck(cudaGraphLaunch(executable,stream));ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));gpu.push_back(ms/iterations);wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/iterations);
  reset();ck(cudaDeviceSynchronize());t=std::chrono::steady_clock::now();ck(cudaEventRecord(start,stream));for(int i=0;i<iterations;i++)launch(stream);ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&ms,start,end));directGpu.push_back(ms/iterations);directWall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/iterations);
 }
 std::cout<<"{\"key\":\""<<key<<"\",\"verified\":true,\"iterations\":"<<iterations<<",\"samples\":9,\"warmup\":5,\"gpuMedianMs\":"<<med(gpu)<<",\"wallMedianMs\":"<<med(wall)<<",\"directGpuMedianMs\":"<<med(directGpu)<<",\"directWallMedianMs\":"<<med(directWall)<<",\"timings\":[";
 for(int s=0;s<9;s++){if(s)std::cout<<',';std::cout<<"{\"gpuMs\":"<<gpu[s]<<",\"wallMs\":"<<wall[s]<<",\"directGpuMs\":"<<directGpu[s]<<",\"directWallMs\":"<<directWall[s]<<'}';}std::cout<<"]}"<<std::endl;
 ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));ck(cudaStreamDestroy(stream));
}
