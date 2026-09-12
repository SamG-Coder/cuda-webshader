template<int N>
__global__ void shared_child(int *out) {
    extern __shared__ int s[];
    unsigned lane=threadIdx.x;
    s[lane]=blockIdx.x*100+lane;
    __syncthreads();
    out[blockIdx.x*N+lane]=s[N-1-lane];
}
__global__ void shared_parent(int *out) {
    const int threads=32;
    const int bytes=threads*sizeof(int);
    if(threadIdx.x==0) shared_child<32><<<2,threads,bytes>>>(out);
}
