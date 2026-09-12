// SPDX-License-Identifier: MIT
__device__ float chain(float& a,int& b){float e=0.f;a=e=1.25f;a=b=3.75f;return e;}
__device__ int colours(const uchar4& a,const uchar4& b){return int(b.x)-int(a.x)+int(b.z)-int(a.z);}
template<class T> __device__ T converted(){return T(4.0)+T(0.125);}
__global__ void referenceEffects(float* out){float a=0.f;int b=0;out[0]=chain(a,b);out[1]=a;out[2]=float(b);int n=4,count=0;while(--n && count++<2){}out[3]=float(n);out[4]=float(count);int i=0;bool skip=false&&++i;bool skip2=true||++i;bool take=true&&++i;out[5]=float(i);out[6]=float(i++);out[7]=float(++i);uchar4 p=make_uchar4(1,2,3,4),q=make_uchar4(7,8,9,10);out[8]=float(colours(p,q));out[9]=float(colours(p,make_uchar4(2,4,6,8)));out[10]=converted<float>();out[11]=(float)0.1;out[12]=float(-0.0);}
