// SPDX-License-Identifier: MIT
__device__ void adjustByte(unsigned char* pixel) {
    pixel[0] += 17;
    pixel[0]++;
    pixel[0] ^= 165;
}
__global__ void bytePixels(const unsigned char* input, unsigned char* output,
                          unsigned int width, unsigned int height, unsigned int pitch) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<width*height){
        unsigned char* pixel=output+((i/width)*pitch+1);
        pixel=&pixel[i%width];
        pixel[0]=input[i];
        adjustByte(pixel);
    }
}
