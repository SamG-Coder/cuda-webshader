// SPDX-License-Identifier: MIT
__constant__ short signedValues[]={-32768,32767};
__constant__ unsigned short unsignedValues[]={65535};
__global__ void roundingProbe(const float*input,float*output){unsigned i=threadIdx.x;if(i<12)output[i]=roundf(input[i]);if(i==0){output[12]=signedValues[0];output[13]=signedValues[1];output[14]=__mul24(17,19);output[15]=unsignedValues[0];}}
