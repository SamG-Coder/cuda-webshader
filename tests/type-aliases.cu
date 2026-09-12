// SPDX-License-Identifier: MIT
typedef unsigned char VolumeType;
typedef VolumeType Byte;
typedef float Scalar;
typedef uchar4 Pixel;
template<typename T> __device__ T twice(T x){return x+x;}
__global__ void aliasProbe(uint* out){Byte b=(Byte)260;Scalar f=Scalar(3);Pixel p=make_uchar4(b,5,6,7);out[0]=b;out[1]=(uint)twice<Scalar>(f);out[2]=p.y;out[3]=4u;}
