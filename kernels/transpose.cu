#define TILE 32
#define ROWS 8
// Launch [32,8,1]. Padding avoids the conventional 32-wide shared-memory bank-conflict pattern.
__global__ void transpose(const float* input, float* output,
                          unsigned int width, unsigned int height) {
    __shared__ float tile[TILE][TILE+1];
    unsigned int x = blockIdx.x*TILE+threadIdx.x;
    unsigned int y = blockIdx.y*TILE+threadIdx.y;
    for (unsigned int j = 0; j<TILE; j+=ROWS) {
        tile[threadIdx.y+j][threadIdx.x] = (x<width && y+j<height) ? input[(y+j)*width+x] : 0.0f;
    }
    __syncthreads();
    x = blockIdx.y*TILE+threadIdx.x;
    y = blockIdx.x*TILE+threadIdx.y;
    for (unsigned int j = 0; j<TILE; j+=ROWS) {
        if (x<height && y+j<width) output[(y+j)*height+x] = tile[threadIdx.x][threadIdx.y+j];
    }
}
