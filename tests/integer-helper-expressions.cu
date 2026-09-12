// MIT compiler probe: the same descending integer-template pattern used by
// NVIDIA convolutionTexture, with an independently checkable arithmetic result.
template<int i> __device__ int descending(int x) {
    return x + i + descending<i - 1>(x);
}
template<> __device__ int descending<-1>(int x) { return 0; }
template<int n> __device__ int forward(int x) { return descending<2 * n>(x); }
template<int n> __device__ int signedValue() { return n; }
__global__ void integerTemplates(const int* input, int* output) {
    output[0] = descending<2 * 8>(input[0]);
    output[1] = descending<-1>(input[0]);
    output[2] = forward<3>(input[1]);
    output[3] = descending<(9 / 2) - 1>(input[1]);
    output[4] = signedValue<(-2147483647 - 1)>();
    output[5] = signedValue<-7 / 2>();
}
