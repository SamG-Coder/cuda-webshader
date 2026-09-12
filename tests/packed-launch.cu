// SPDX-License-Identifier: MIT
__device__ uchar4 pickColour(uchar4 colour,bool choose){return choose?colour:make_uchar4(colour.w,colour.z,colour.y,colour.x);}
__global__ void packedLaunch(uchar4* output,uchar4 colour,bool choose,unsigned int slot){output[slot]=pickColour(colour,choose);}
