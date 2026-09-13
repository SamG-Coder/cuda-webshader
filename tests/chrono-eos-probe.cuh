// MIT diagnostic adapter around unchanged Chrono Eos.
__global__ void eosProbe(const Real4* input, float* output, unsigned n) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 Real gama=7;Real q=input[i].x/paramsD.rho0;
 output[i*4]=q;output[i*4+1]=pow(q,gama);
 output[i*4+2]=Eos(input[i].x,paramsD.eos_type);
 output[i*4+3]=paramsD.rho0*paramsD.Cs*paramsD.Cs/gama;
}
