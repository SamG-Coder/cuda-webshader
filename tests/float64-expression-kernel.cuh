// SPDX-License-Identifier: MIT
__global__ void expressions(const float* a,const float* b,float* out,unsigned int count){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=count)return;
 out[i*6]=(a[i]+0.0)+b[i];
 out[i*6+1]=(a[i]+0.0)*b[i];
 out[i*6+2]=(a[i]+0.0)/b[i];
 out[i*6+3]=(a[i]+0.0)<(a[i]+b[i]*0.5);
 float v=a[i];v+=b[i]*1.0;out[i*6+4]=v;
 out[i*6+5]=((a[i]+0.0)+b[i])-a[i];
}
