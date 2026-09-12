// SPDX-License-Identifier: MIT
// Exercise the 1D float4 surface ABI needed by NVIDIA volume pre-integration.
__global__ void writeTransfer(float4* input, cudaSurfaceObject_t output, unsigned int count) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < count) surf1Dwrite(input[x], output, x * sizeof(float4));
}
__global__ void readTransfer(float4* output, cudaTextureObject_t input, unsigned int count) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < count) output[x] = tex1D<float4>(input, (float(x) + 0.5f) / float(count));
}
__global__ void readTransferZero(float4* output, cudaTextureObject_t input) {
    output[0] = tex1D<float4>(input, 0);
}
