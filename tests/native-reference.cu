// OPTIONAL native CUDA correctness baseline. Requires the NVIDIA CUDA Toolkit + a CUDA GPU.
// Not used by the browser, and not executed in the authoring environment.
// nvcc -O3 -std=c++17 -arch=native tests/native-reference.cu -o native-reference
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <string>
#include "../kernels/saxpy.cu"
#include "../kernels/matmul_naive.cu"
#include "../kernels/matmul_tiled.cu"
#include "../kernels/matmul_register.cu"
#include "../kernels/reduce_sum.cu"

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct DeviceBuffer {
    float* ptr = nullptr;
    size_t count;
    explicit DeviceBuffer(size_t n) : count(n) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), (n ? n : 1) * sizeof(float))); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<float>& values) {
        if (values.size() != count) throw std::runtime_error("Upload size mismatch");
        if (count) check(cudaMemcpy(ptr, values.data(), count*sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> read() const {
        std::vector<float> values(count);
        if (count) check(cudaMemcpy(values.data(), ptr, count*sizeof(float), cudaMemcpyDeviceToHost));
        return values;
    }
};
static void close(const std::vector<float>& actual, const std::vector<float>& expected, const char* name) {
    if (actual.size() != expected.size()) throw std::runtime_error("Output length mismatch");
    double maxError = 0;
    for (size_t i=0; i<actual.size(); ++i) {
        double error = std::fabs(static_cast<double>(actual[i])-expected[i]);
        if (!std::isfinite(actual[i]) || error > 0.001+0.001*std::fabs(expected[i]))
            throw std::runtime_error(std::string(name)+" mismatch at "+std::to_string(i));
        maxError = std::fmax(maxError,error);
    }
    std::cout << "PASS " << name << " max_abs_error=" << maxError << '\n';
}
int main() {
    try {
        cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties,0));
        std::cout << "NATIVE CUDA CORRECTNESS ONLY; GPU=" << properties.name << '\n';
        const unsigned n=1031;
        std::vector<float> x(n), y(n), expected(n);
        for (unsigned i=0;i<n;++i) { x[i]=(int(i%31)-15)*0.125f; y[i]=(int(i%17)-8)*0.25f; expected[i]=std::fma(0.75f,x[i],y[i]); }
        DeviceBuffer dx(n),dy(n); dx.upload(x); dy.upload(y);
        saxpy<<<(n+127)/128,128>>>(dx.ptr,dy.ptr,0.75f,n); check(cudaGetLastError());
        close(dy.read(),expected,"SAXPY tail 1031");

        const unsigned M=17,N=19,K=23;
        std::vector<float> A(M*K),B(K*N),C(M*N,0);
        for (unsigned i=0;i<A.size();++i) A[i]=(int(i%29)-14)*0.0625f;
        for (unsigned i=0;i<B.size();++i) B[i]=(int(i%19)-9)*0.125f;
        for (unsigned r=0;r<M;++r) for(unsigned c=0;c<N;++c)
            for(unsigned k=0;k<K;++k) C[r*N+c]=std::fma(A[r*K+k],B[k*N+c],C[r*N+c]);
        DeviceBuffer da(A.size()),db(B.size()),dc(C.size()); da.upload(A);db.upload(B);
        matmul_naive<<<dim3((N+7)/8,(M+7)/8),dim3(8,8)>>>(da.ptr,db.ptr,dc.ptr,M,N,K);check(cudaGetLastError());
        close(dc.read(),C,"Matrix baseline 17x19x23");
        matmul_tiled<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(da.ptr,db.ptr,dc.ptr,M,N,K);check(cudaGetLastError());
        close(dc.read(),C,"Matrix shared tile 17x19x23");
        matmul_register<<<dim3((N+15)/16,(M+15)/16),dim3(8,8)>>>(da.ptr,db.ptr,dc.ptr,M,N,K);check(cudaGetLastError());
        close(dc.read(),C,"Matrix register tile 17x19x23");

        unsigned length=4099; std::vector<float> values(length);float sum=0;
        for(unsigned i=0;i<length;++i) {values[i]=(int(i%13)-6)*0.125f;sum+=values[i];}
        std::vector<std::unique_ptr<DeviceBuffer>> levels;
        levels.emplace_back(std::make_unique<DeviceBuffer>(length));levels.back()->upload(values);
        while(length>1){unsigned groups=(length+255)/256;float* input=levels.back()->ptr;
            levels.emplace_back(std::make_unique<DeviceBuffer>(groups));
            reduce_sum<<<groups,128>>>(input,levels.back()->ptr,length);check(cudaGetLastError());length=groups;}
        close(levels.back()->read(),std::vector<float>{sum},"Hierarchical reduction 4099");
        check(cudaDeviceSynchronize());
        std::cout << "5/5 native CUDA checks passed. This is NOT a CUDA-versus-WebGPU speed comparison.\n";
        return 0;
    } catch(const std::exception& error) {std::cerr << "FAIL " << error.what() << '\n';return 1;}
}
