// Private histogram per workgroup: global atomics only merge the 256 local bins.
// The caller must clear bins before each independent histogram.
__global__ void histogram(const unsigned int* input, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int localBins[256];
    for (unsigned int b = threadIdx.x; b<256; b+=blockDim.x) localBins[b] = 0u;
    __syncthreads();
    for (unsigned int i = blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x) {
        atomicAdd(&localBins[input[i]&255u],1u);
    }
    __syncthreads();
    for (unsigned int b = threadIdx.x; b<256; b+=blockDim.x) {
        unsigned int value = localBins[b];
        if (value>0u) atomicAdd(&bins[b],value);
    }
}
