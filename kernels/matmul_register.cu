#define TILE 16
#define HALF_TILE 8
// Launch [8,8,1]. Each invocation computes 2x2 outputs from a 16x16 shared tile.
__global__ void matmul_register(const float* A, const float* B, float* C,
                                unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row0 = blockIdx.y*TILE+ty;
    unsigned int row1 = row0+HALF_TILE;
    unsigned int col0 = blockIdx.x*TILE+tx;
    unsigned int col1 = col0+HALF_TILE;
    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;
    for (unsigned int base = 0; base < K; base += TILE) {
        tileA[ty][tx] = (row0<M && base+tx<K) ? A[row0*K+base+tx] : 0.0f;
        tileA[ty][tx+HALF_TILE] = (row0<M && base+tx+HALF_TILE<K) ? A[row0*K+base+tx+HALF_TILE] : 0.0f;
        tileA[ty+HALF_TILE][tx] = (row1<M && base+tx<K) ? A[row1*K+base+tx] : 0.0f;
        tileA[ty+HALF_TILE][tx+HALF_TILE] = (row1<M && base+tx+HALF_TILE<K) ? A[row1*K+base+tx+HALF_TILE] : 0.0f;
        tileB[ty][tx] = (base+ty<K && col0<N) ? B[(base+ty)*N+col0] : 0.0f;
        tileB[ty][tx+HALF_TILE] = (base+ty<K && col1<N) ? B[(base+ty)*N+col1] : 0.0f;
        tileB[ty+HALF_TILE][tx] = (base+ty+HALF_TILE<K && col0<N) ? B[(base+ty+HALF_TILE)*N+col0] : 0.0f;
        tileB[ty+HALF_TILE][tx+HALF_TILE] = (base+ty+HALF_TILE<K && col1<N) ? B[(base+ty+HALF_TILE)*N+col1] : 0.0f;
        __syncthreads();
        for (unsigned int k = 0; k < TILE; ++k) {
            float a0 = tileA[ty][k];
            float a1 = tileA[ty+HALF_TILE][k];
            float b0 = tileB[k][tx];
            float b1 = tileB[k][tx+HALF_TILE];
            c00 = fmaf(a0,b0,c00);
            c01 = fmaf(a0,b1,c01);
            c10 = fmaf(a1,b0,c10);
            c11 = fmaf(a1,b1,c11);
        }
        __syncthreads();
    }
    if (row0<M && col0<N) C[row0*N+col0] = c00;
    if (row0<M && col1<N) C[row0*N+col1] = c01;
    if (row1<M && col0<N) C[row1*N+col0] = c10;
    if (row1<M && col1<N) C[row1*N+col1] = c11;
}
