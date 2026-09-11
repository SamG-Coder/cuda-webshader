#define TILE 16
// Launch exactly [TILE,TILE,1]. Edge tiles are zero-padded, never early-returned.
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y*TILE+ty;
    unsigned int col = blockIdx.x*TILE+tx;
    float sum = 0.0f;
    for (unsigned int base = 0; base < K; base += TILE) {
        tileA[ty][tx] = (row < M && base+tx < K) ? A[row*K+base+tx] : 0.0f;
        tileB[ty][tx] = (base+ty < K && col < N) ? B[(base+ty)*N+col] : 0.0f;
        __syncthreads();
        for (unsigned int k = 0; k < TILE; ++k) sum = fmaf(tileA[ty][k], tileB[k][tx], sum);
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}
