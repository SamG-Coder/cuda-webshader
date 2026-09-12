__global__ void offset_child(int *out,int value){out[threadIdx.x]=value*10+threadIdx.x;}
__global__ void offset_parent(int *out){
 out+=1;
 int *children=out+1;
 int i=threadIdx.x;
 int child_offset=i*4;
 offset_child<<<1,4>>>(&children[child_offset],i);
 children+=1;
}
__global__ void invalid_offset(int *out,int offset){offset_child<<<1,4>>>(out+offset,7);}
