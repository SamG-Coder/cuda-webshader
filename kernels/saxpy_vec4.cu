// Four elements per invocation, 16-byte CUDA/WGSL-compatible float4 layout.
// n4 is the count of float4 records. Handle any scalar tail separately.
__global__ void saxpy_vec4(const float4* __restrict__ x, float4* __restrict__ y,
                           float a, unsigned int n4) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 xv = x[i];
        float4 yv = y[i];
        y[i] = make_float4(fmaf(a, xv.x, yv.x), fmaf(a, xv.y, yv.y),
                          fmaf(a, xv.z, yv.z), fmaf(a, xv.w, yv.w));
    }
}
