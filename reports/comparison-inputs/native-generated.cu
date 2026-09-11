#include "../../benchmarks/native-support.cuh"
int main(){try{std::cout<<std::setprecision(10);cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));std::cerr<<"GPU: "<<prop.name<<" SM "<<prop.major<<"."<<prop.minor<<"\n";
{
Buffer x("reports/comparison-inputs/small-saxpy-x.bin");
Buffer y("reports/comparison-inputs/small-saxpy-y.bin");
auto launch=[&](cudaStream_t stream){saxpy<<<dim3(2048),dim3(128,1,1),0,stream>>>((float*)x.p,(float*)y.p,0.1250000000f,262144u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
y.verify("reports/comparison-inputs/small-saxpy-y-expected.bin",false);
benchmark("small-saxpy",2048,launch,[&](){x.reset();y.reset();});
}
{
Buffer x("reports/comparison-inputs/small-saxpy_vec4-x.bin");
Buffer y("reports/comparison-inputs/small-saxpy_vec4-y.bin");
auto launch=[&](cudaStream_t stream){saxpy_vec4<<<dim3(512),dim3(128,1,1),0,stream>>>((float4*)x.p,(float4*)y.p,0.1250000000f,65536u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
y.verify("reports/comparison-inputs/small-saxpy_vec4-y-expected.bin",false);
benchmark("small-saxpy_vec4",2048,launch,[&](){x.reset();y.reset();});
}
{
Buffer A("reports/comparison-inputs/small-matmul_naive-A.bin");
Buffer B("reports/comparison-inputs/small-matmul_naive-B.bin");
Buffer C("reports/comparison-inputs/small-matmul_naive-C.bin");
auto launch=[&](cudaStream_t stream){matmul_naive<<<dim3(16,16),dim3(8,8,1),0,stream>>>((float*)A.p,(float*)B.p,(float*)C.p,128u,128u,128u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
C.verify("reports/comparison-inputs/small-matmul_naive-C-expected.bin",false);
benchmark("small-matmul_naive",512,launch,[&](){A.reset();B.reset();C.reset();});
}
{
Buffer A("reports/comparison-inputs/small-matmul_tiled-A.bin");
Buffer B("reports/comparison-inputs/small-matmul_tiled-B.bin");
Buffer C("reports/comparison-inputs/small-matmul_tiled-C.bin");
auto launch=[&](cudaStream_t stream){matmul_tiled<<<dim3(8,8),dim3(16,16,1),0,stream>>>((float*)A.p,(float*)B.p,(float*)C.p,128u,128u,128u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
C.verify("reports/comparison-inputs/small-matmul_tiled-C-expected.bin",false);
benchmark("small-matmul_tiled",512,launch,[&](){A.reset();B.reset();C.reset();});
}
{
Buffer A("reports/comparison-inputs/small-matmul_register-A.bin");
Buffer B("reports/comparison-inputs/small-matmul_register-B.bin");
Buffer C("reports/comparison-inputs/small-matmul_register-C.bin");
auto launch=[&](cudaStream_t stream){matmul_register<<<dim3(8,8),dim3(8,8,1),0,stream>>>((float*)A.p,(float*)B.p,(float*)C.p,128u,128u,128u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
C.verify("reports/comparison-inputs/small-matmul_register-C-expected.bin",false);
benchmark("small-matmul_register",512,launch,[&](){A.reset();B.reset();C.reset();});
}
{
Buffer input("reports/comparison-inputs/small-reduce_sum-input.bin");
Buffer level0(4096);
Buffer level1(16);
Buffer level2(4);
Buffer& output=level2;
auto launch=[&](cudaStream_t stream){reduce_sum<<<1024,128,0,stream>>>((float*)input.p,(float*)level0.p,262144u);reduce_sum<<<4,128,0,stream>>>((float*)level0.p,(float*)level1.p,1024u);reduce_sum<<<1,128,0,stream>>>((float*)level1.p,(float*)level2.p,4u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
output.verify("reports/comparison-inputs/small-reduce_sum-output-expected.bin",false);
benchmark("small-reduce_sum",2048,launch,[&](){input.reset();});
}
{
Buffer input("reports/comparison-inputs/small-convolution-input.bin");
Buffer output("reports/comparison-inputs/small-convolution-output.bin");
auto launch=[&](cudaStream_t stream){convolution<<<dim3(2048),dim3(128,1,1),0,stream>>>((float*)input.p,(float*)output.p,262144u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
output.verify("reports/comparison-inputs/small-convolution-output-expected.bin",false);
benchmark("small-convolution",2048,launch,[&](){input.reset();output.reset();});
}
{
Buffer input("reports/comparison-inputs/small-histogram-input.bin");
Buffer bins("reports/comparison-inputs/small-histogram-bins.bin");
auto launch=[&](cudaStream_t stream){clear_bins<<<1,256,0,stream>>>((unsigned int*)bins.p);histogram<<<dim3(256),dim3(128,1,1),0,stream>>>((unsigned int*)input.p,(unsigned int*)bins.p,262144u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
bins.verify("reports/comparison-inputs/small-histogram-bins-expected.bin",true);
benchmark("small-histogram",2048,launch,[&](){input.reset();bins.reset();});
}
{
Buffer input("reports/comparison-inputs/small-transpose-input.bin");
Buffer output("reports/comparison-inputs/small-transpose-output.bin");
auto launch=[&](cudaStream_t stream){transpose<<<dim3(16,16),dim3(32,8,1),0,stream>>>((float*)input.p,(float*)output.p,512u,512u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
output.verify("reports/comparison-inputs/small-transpose-output-expected.bin",false);
benchmark("small-transpose",2048,launch,[&](){input.reset();output.reset();});
}
{
Buffer position("reports/comparison-inputs/small-particles-position.bin");
Buffer velocity("reports/comparison-inputs/small-particles-velocity.bin");
auto launch=[&](cudaStream_t stream){particles<<<dim3(512),dim3(128,1,1),0,stream>>>((float4*)position.p,(float4*)velocity.p,65536u,0.0160000008f,2.5000000000f,1.0000000000f);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
position.verify("reports/comparison-inputs/small-particles-position-expected.bin",false);
velocity.verify("reports/comparison-inputs/small-particles-velocity-expected.bin",false);
benchmark("small-particles",2048,launch,[&](){position.reset();velocity.reset();});
}
{
Buffer x("reports/comparison-inputs/large-saxpy-x.bin");
Buffer y("reports/comparison-inputs/large-saxpy-y.bin");
auto launch=[&](cudaStream_t stream){saxpy<<<dim3(32768),dim3(128,1,1),0,stream>>>((float*)x.p,(float*)y.p,0.1250000000f,4194304u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
y.verify("reports/comparison-inputs/large-saxpy-y-expected.bin",false);
benchmark("large-saxpy",2048,launch,[&](){x.reset();y.reset();});
}
{
Buffer x("reports/comparison-inputs/large-saxpy_vec4-x.bin");
Buffer y("reports/comparison-inputs/large-saxpy_vec4-y.bin");
auto launch=[&](cudaStream_t stream){saxpy_vec4<<<dim3(8192),dim3(128,1,1),0,stream>>>((float4*)x.p,(float4*)y.p,0.1250000000f,1048576u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
y.verify("reports/comparison-inputs/large-saxpy_vec4-y-expected.bin",false);
benchmark("large-saxpy_vec4",2048,launch,[&](){x.reset();y.reset();});
}
{
Buffer A("reports/comparison-inputs/large-matmul_naive-A.bin");
Buffer B("reports/comparison-inputs/large-matmul_naive-B.bin");
Buffer C("reports/comparison-inputs/large-matmul_naive-C.bin");
auto launch=[&](cudaStream_t stream){matmul_naive<<<dim3(64,64),dim3(8,8,1),0,stream>>>((float*)A.p,(float*)B.p,(float*)C.p,512u,512u,512u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
C.verify("reports/comparison-inputs/large-matmul_naive-C-expected.bin",false);
benchmark("large-matmul_naive",512,launch,[&](){A.reset();B.reset();C.reset();});
}
{
Buffer A("reports/comparison-inputs/large-matmul_tiled-A.bin");
Buffer B("reports/comparison-inputs/large-matmul_tiled-B.bin");
Buffer C("reports/comparison-inputs/large-matmul_tiled-C.bin");
auto launch=[&](cudaStream_t stream){matmul_tiled<<<dim3(32,32),dim3(16,16,1),0,stream>>>((float*)A.p,(float*)B.p,(float*)C.p,512u,512u,512u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
C.verify("reports/comparison-inputs/large-matmul_tiled-C-expected.bin",false);
benchmark("large-matmul_tiled",512,launch,[&](){A.reset();B.reset();C.reset();});
}
{
Buffer A("reports/comparison-inputs/large-matmul_register-A.bin");
Buffer B("reports/comparison-inputs/large-matmul_register-B.bin");
Buffer C("reports/comparison-inputs/large-matmul_register-C.bin");
auto launch=[&](cudaStream_t stream){matmul_register<<<dim3(32,32),dim3(8,8,1),0,stream>>>((float*)A.p,(float*)B.p,(float*)C.p,512u,512u,512u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
C.verify("reports/comparison-inputs/large-matmul_register-C-expected.bin",false);
benchmark("large-matmul_register",512,launch,[&](){A.reset();B.reset();C.reset();});
}
{
Buffer input("reports/comparison-inputs/large-reduce_sum-input.bin");
Buffer level0(65536);
Buffer level1(256);
Buffer level2(4);
Buffer& output=level2;
auto launch=[&](cudaStream_t stream){reduce_sum<<<16384,128,0,stream>>>((float*)input.p,(float*)level0.p,4194304u);reduce_sum<<<64,128,0,stream>>>((float*)level0.p,(float*)level1.p,16384u);reduce_sum<<<1,128,0,stream>>>((float*)level1.p,(float*)level2.p,64u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
output.verify("reports/comparison-inputs/large-reduce_sum-output-expected.bin",false);
benchmark("large-reduce_sum",2048,launch,[&](){input.reset();});
}
{
Buffer input("reports/comparison-inputs/large-convolution-input.bin");
Buffer output("reports/comparison-inputs/large-convolution-output.bin");
auto launch=[&](cudaStream_t stream){convolution<<<dim3(32768),dim3(128,1,1),0,stream>>>((float*)input.p,(float*)output.p,4194304u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
output.verify("reports/comparison-inputs/large-convolution-output-expected.bin",false);
benchmark("large-convolution",2048,launch,[&](){input.reset();output.reset();});
}
{
Buffer input("reports/comparison-inputs/large-histogram-input.bin");
Buffer bins("reports/comparison-inputs/large-histogram-bins.bin");
auto launch=[&](cudaStream_t stream){clear_bins<<<1,256,0,stream>>>((unsigned int*)bins.p);histogram<<<dim3(256),dim3(128,1,1),0,stream>>>((unsigned int*)input.p,(unsigned int*)bins.p,4194304u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
bins.verify("reports/comparison-inputs/large-histogram-bins-expected.bin",true);
benchmark("large-histogram",2048,launch,[&](){input.reset();bins.reset();});
}
{
Buffer input("reports/comparison-inputs/large-transpose-input.bin");
Buffer output("reports/comparison-inputs/large-transpose-output.bin");
auto launch=[&](cudaStream_t stream){transpose<<<dim3(64,64),dim3(32,8,1),0,stream>>>((float*)input.p,(float*)output.p,2048u,2048u);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
output.verify("reports/comparison-inputs/large-transpose-output-expected.bin",false);
benchmark("large-transpose",2048,launch,[&](){input.reset();output.reset();});
}
{
Buffer position("reports/comparison-inputs/large-particles-position.bin");
Buffer velocity("reports/comparison-inputs/large-particles-velocity.bin");
auto launch=[&](cudaStream_t stream){particles<<<dim3(2048),dim3(128,1,1),0,stream>>>((float4*)position.p,(float4*)velocity.p,262144u,0.0160000008f,2.5000000000f,1.0000000000f);ck(cudaGetLastError());};
launch(nullptr);ck(cudaDeviceSynchronize());
position.verify("reports/comparison-inputs/large-particles-position-expected.bin",false);
velocity.verify("reports/comparison-inputs/large-particles-velocity-expected.bin",false);
benchmark("large-particles",2048,launch,[&](){position.reset();velocity.reset();});
}
return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
