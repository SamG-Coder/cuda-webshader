// SPDX-License-Identifier: MIT
// Stable unsigned key/value ordering by carrying the original position as a tie-breaker.
export const SORT_SOURCE=String.raw`
__global__ void sortPrepare(const uint*keys,const uint*values,uint2*pairs,uint*order,uint count,uint padded){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<padded){pairs[i]=i<count?make_uint2(keys[i],values[i]):make_uint2(0xffffffffu,0u);order[i]=i;}}
__global__ void sortStage(uint2*pairs,uint*order,uint count,uint size,uint stride){uint i=blockIdx.x*blockDim.x+threadIdx.x,j=i^stride;if(i<count&&j>i&&j<count){uint2 a=pairs[i],b=pairs[j];uint ai=order[i],bi=order[j];bool greater=a.x>b.x||(a.x==b.x&&ai>bi);bool ascending=(i&size)==0;if(greater==ascending){pairs[i]=b;pairs[j]=a;order[i]=bi;order[j]=ai;}}}
__global__ void sortFinish(const uint2*pairs,uint*keys,uint*values,uint count){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){uint2 p=pairs[i];keys[i]=p.x;values[i]=p.y;}}
`;
