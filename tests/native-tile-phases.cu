namespace cg=cooperative_groups;
__global__ void tile_phases(int*out){
 cg::thread_block cta=cg::this_thread_block();
 cg::thread_block_tile<32> tile=cg::tiled_partition<32>(cta);
 const int warp=threadIdx.x/warpSize;
 __shared__ int values[16];
 if(threadIdx.x<16)values[threadIdx.x]=threadIdx.x+1;
 cg::sync(cta);
 if(warp==0){
  int sum=values[3];
  for(int row=1;row<4;++row){
   int tmp=values[row*4+3];
   cg::sync(tile);
   if(tile.thread_rank()<4)values[row*4+tile.thread_rank()]+=sum;
   cg::sync(tile);
   sum+=tmp;
  }
 }
 cg::sync(cta);
 out[threadIdx.x]=values[threadIdx.x%16];
}
