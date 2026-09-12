// SPDX-License-Identifier: MIT
__device__ void twiddle(float2 &v,float phase){__sincosf(phase,&v.y,&v.x);}
__global__ void sincosProbe(float*phases,int*bits,float4*output,int*first,unsigned int n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float2 v;twiddle(v,phases[i]);float s=phases[i],c;sincosf(s,&s,&c);output[i]=make_float4(v.y,v.x,s,c);first[i]=__ffs(bits[i]);}}
