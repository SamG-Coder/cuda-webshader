// MIT probe for the paired volume and transfer textures used by volumeRender.
__global__ void transferProbe(const float* coordinates, float4* output,
                              cudaTextureObject_t volume, cudaTextureObject_t transfer,
                              float shift, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value = tex3D<float>(volume, 0.5f, 0.5f, 0.5f);
        output[i] = tex1D<float4>(transfer, coordinates[i] + value * shift);
    }
}
