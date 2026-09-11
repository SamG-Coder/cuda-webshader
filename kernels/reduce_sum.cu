#define BLOCK_SIZE 128
// Launch [BLOCK_SIZE,1,1]. Two inputs per lane; recursively reduce partials in separate dispatches.
// There is intentionally NO attempt at a grid-wide barrier in a single dispatch.
__global__ void reduce_sum(const float* input, float* output, unsigned int n) {
    __shared__ float partial[BLOCK_SIZE];
    unsigned int lane = threadIdx.x;
    unsigned int i = blockIdx.x*BLOCK_SIZE*2+lane;
    float sum = 0.0f;
    if (i<n) sum = input[i];
    if (i+BLOCK_SIZE<n) sum += input[i+BLOCK_SIZE];
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = BLOCK_SIZE/2; stride>0; stride >>= 1) {
        if (lane<stride) partial[lane] += partial[lane+stride];
        __syncthreads();
    }
    if (lane==0) output[blockIdx.x] = partial[0];
}
