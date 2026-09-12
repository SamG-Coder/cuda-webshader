__global__ void array_init(int*out){
 int i=threadIdx.x;
 int a[4]={i++,i++,i++};
 int b[2][3]={{i,7},{}};
 const int c[3]={9};
 for(int j=0;j<4;j++)out[threadIdx.x*10+j]=a[j];
 out[threadIdx.x*10+4]=i;
 out[threadIdx.x*10+5]=b[0][0];
 out[threadIdx.x*10+6]=b[0][2];
 out[threadIdx.x*10+7]=b[1][1];
 out[threadIdx.x*10+8]=c[0];
 out[threadIdx.x*10+9]=max(warpSize,c[2]);
}
