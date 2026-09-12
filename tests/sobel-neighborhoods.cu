// SPDX-License-Identifier: MIT — calls the unchanged NVIDIA ComputeSobel helper.
__global__ void sobelNeighborhoods(const unsigned char* input, unsigned char* output,
                                   unsigned int n, float scale) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){unsigned int p=i*9;output[i]=ComputeSobel(input[p],input[p+1],input[p+2],input[p+3],input[p+4],input[p+5],input[p+6],input[p+7],input[p+8],scale);}
}
