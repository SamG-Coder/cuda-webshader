__global__ void recursive_steps(unsigned*out,unsigned depth,unsigned limit,unsigned fanout){
 out[depth*32+threadIdx.x]=depth+1;
 __syncthreads();
 if(depth<limit&&threadIdx.x<fanout)recursive_steps<<<1,32>>>(out,depth+1,limit,fanout);
}
