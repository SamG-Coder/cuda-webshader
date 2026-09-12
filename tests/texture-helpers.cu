// MIT probe for explicit texture/sampler arguments in nested device helpers.
template<int i> __device__ float accumulate(float u, float v, cudaTextureObject_t image) {
    return tex2D<float>(image, u, v) * (float)(i + 1) + accumulate<i - 1>(u, v, image);
}
template<> __device__ float accumulate<-1>(float u, float v, cudaTextureObject_t image) { return 0.0f; }
__device__ float route(cudaTextureObject_t image, float u, float v) {
    return accumulate<3>(u, v, image);
}
__global__ void textureHelpers(float* output, cudaTextureObject_t first, cudaTextureObject_t second) {
    unsigned int i = threadIdx.x;
    if(i < 4) {
        float u = ((float)i + 0.25f) / 4.0f;
        output[i] = route(first, u, 0.5f) + 2.0f * route(second, u, 0.5f);
        output[i + 4] = route(second, u, 0.5f) + 2.0f * route(first, u, 0.5f);
    }
}
