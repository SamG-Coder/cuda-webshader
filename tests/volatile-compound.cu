__global__ void volatile_compound(int*out,float*fout){
 __shared__ int storage[64];
 volatile int *rows[2];
 rows[0]=(volatile int*)&storage[0];
 rows[1]=(volatile int*)&storage[32];
 volatile __shared__ float floats[32];
 unsigned lane=threadIdx.x;
 rows[0][lane]=lane;
 rows[1][lane]=100+lane;
 floats[lane]=float(lane)+1.f;
 __syncthreads();
 for(int row=0;row<2;++row){
  rows[row][lane]+=7;
  rows[row][lane]*=3;
  rows[row][lane]>>=1;
  rows[row][lane]^=5;
 }
 floats[lane]*=2.f;
 floats[lane]/=4.f;
 __syncthreads();
 out[lane*2]=rows[0][(lane+1)%32];
 out[lane*2+1]=rows[1][(lane+1)%32];
 fout[lane]=floats[(lane+1)%32];
}
