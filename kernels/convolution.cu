#define BLOCK_SIZE 128
// Five-tap Gaussian-like stencil, zero outside the signal. Halo staged once per block.
__global__ void convolution(const float* input, float* output, unsigned int n) {
    __shared__ float tile[BLOCK_SIZE+4];
    unsigned int lane = threadIdx.x;
    unsigned int i = blockIdx.x*BLOCK_SIZE+lane;
    tile[lane+2] = i<n ? input[i] : 0.0f;
    if (lane<2) {
        int left = (int)(blockIdx.x*BLOCK_SIZE+lane)-2;
        unsigned int right = blockIdx.x*BLOCK_SIZE+BLOCK_SIZE+lane;
        tile[lane] = (left>=0 && left<(int)n) ? input[left] : 0.0f;
        tile[BLOCK_SIZE+2+lane] = right<n ? input[right] : 0.0f;
    }
    __syncthreads();
    if (i<n) {
        float sum = tile[lane]*0.0625f;
        sum = fmaf(tile[lane+1],0.25f,sum);
        sum = fmaf(tile[lane+2],0.375f,sum);
        sum = fmaf(tile[lane+3],0.25f,sum);
        output[i] = fmaf(tile[lane+4],0.0625f,sum);
    }
}
