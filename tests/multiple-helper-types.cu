// MIT compiler validation probe, not a replacement NVIDIA showcase.
template<class T, class R> __device__ R converted(T x) { return (R)x + (R)1; }
template<> __device__ float converted<int, float>(int x) { return (float)(x * 10); }
template<class A, class B> __device__ float nested(A a, B b) { return converted<B, A>(b) + converted<A, B>(a); }
template<class A, class B, class C, class D> __device__ D four(A a, B b, C c) { return (D)a + (D)b + (D)c; }
template<class Tag, class R> __device__ R sampled(cudaTextureObject_t tex, float x, float y) { return tex2D<R>(tex, x, y); }
template<class A, class B> __device__ B sampleNested(cudaTextureObject_t tex, float x, float y) { return sampled<A, B>(tex, x, y); }
__global__ void multipleTypes(const float* input, float* output, cudaTextureObject_t tex) {
 output[0] = converted<float, int>(input[0]);
 output[1] = converted<int, float>((int)input[1]);
 output[2] = nested<float, int>(input[0], (int)input[1]);
 output[3] = four<int, float, uint, float>(-2, input[0], 7u);
 output[4] = sampleNested<uint, float>(tex, 1.5f, 0.5f);
 output[5] = sampled<int, float>(tex, 2.0f, 1.0f);
}
