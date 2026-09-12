// SPDX-License-Identifier: MIT
__global__ void shortValues(short a,unsigned short b,int* out) {
  short s=a; unsigned short u=b;
  s+=3; u+=3; s++; u++;
  out[0]=s; out[1]=u; out[2]=a+b;
  out[3]=sizeof(short); out[4]=sizeof(unsigned short);
  out[5]=abs((int)a); out[6]=(short)65535; out[7]=(unsigned short)-1;
}
