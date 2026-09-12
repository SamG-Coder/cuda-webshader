// Exercises the shared pointer layout used by NVIDIA cdpQuadtree.
__global__ void volatile_slots(int *out) {
    extern __shared__ int smem[];
    volatile int *s_num_pts[4];
    for (int i=0; i<4; ++i)
        s_num_pts[i] = (volatile int *)&smem[i*32];
    unsigned lane=threadIdx.x;
    for (int q=0; q<4; ++q) s_num_pts[q][lane]=q*100+lane;
    __syncthreads();
    for (int q=0; q<4; ++q) {
        int x=s_num_pts[q][(lane+1)%32];
        out[lane*4+q]=x;
    }
}
