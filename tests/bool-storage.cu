// MIT: byte-stride CUDA bool arrays, including volatile flag storage.
__global__ void boolStorage(const bool* input, volatile bool* output, volatile bool* flag, int* observed, unsigned int n) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 output[i+3]=!input[i];
 observed[i]=output[i+3]?7:-3;
 if(i==0)*flag=true;
}
