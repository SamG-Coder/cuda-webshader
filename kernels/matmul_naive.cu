// Row-major A[M,K] * B[K,N] -> C[M,N]. Correctness/performance baseline.
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             unsigned int M, unsigned int N, unsigned int K) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; ++k) sum = fmaf(A[row*K+k], B[k*N+col], sum);
        C[row*N+col] = sum;
    }
}
