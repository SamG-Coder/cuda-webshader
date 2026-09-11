// A single fused pass: y = a*x + y. Contiguous, coalesced accesses.
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y,
                      float a, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaf(a, x[i], y[i]);
}
