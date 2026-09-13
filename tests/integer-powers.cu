// MIT compiler regression for runtime integer exponents.
__global__ void integerPowers(const float* input, float* output, unsigned n) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)output[i]=powf(input[i*2],input[i*2+1]);
}
