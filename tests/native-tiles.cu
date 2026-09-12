namespace cg=cooperative_groups;
__global__ void native_tiles(int*out){
 cg::thread_block cta=cg::this_thread_block();
 cg::thread_block_tile<32> tile=cg::tiled_partition<32>(cta);
 int lane=tile.thread_rank();int warp=threadIdx.x/warpSize;
 unsigned mask=tile.ballot(lane<warp+1);
 int prefix=1;
 for(int offset=1;offset<32;offset*=2){int previous=tile.shfl_up(prefix,offset);if(lane>=offset)prefix+=previous;}
 int iterations=0;
 for(int i=lane;tile.any(i<17+warp*25);i+=32)iterations++;
 out[threadIdx.x*5]=__popc(mask);
 out[threadIdx.x*5+1]=tile.shfl(warp*100+lane,7);
 out[threadIdx.x*5+2]=prefix;
 out[threadIdx.x*5+3]=iterations;
 out[threadIdx.x*5+4]=lane;
}
