// SPDX-License-Identifier: MIT
__global__ void sizeValues(unsigned* words,float* values,size_t pitch,int row){
 size_t bytes=pitch*row;
 size_t next=bytes+pitch;
 size_t maximum=(size_t)-1;
 words[0]=(unsigned)bytes;
 words[1]=bytes>0xffffffffu;
 words[2]=(unsigned)next;
 words[3]=maximum>next;
 words[4]=(unsigned)sizeof(size_t);
 values[0]=(float)bytes;
 values[1]=(float)next;
}
