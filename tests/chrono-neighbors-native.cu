// Native reference harness. The included Chrono kernel bodies are unchanged.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "chrono-neighbors.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
static void array(const std::vector<unsigned>& v){printf("[");for(size_t i=0;i<v.size();i++)printf("%s%u",i?",":"",v[i]);printf("]");}
int main(){
 printf("{\"cases\":[");bool first=true;
 for(unsigned n:{0u,1u,31u,127u,128u,129u,255u,256u,257u,1027u}){
  unsigned count=n?n:1,cells=(count+6)/7+2;
  std::vector<unsigned> hashes(count),indices(count),start(cells),end(cells),map(count+4,0xdeadbeefu);
  for(unsigned i=0;i<count;i++){hashes[i]=i/7;indices[i]=n-1-i;}
  unsigned *dh,*di,*ds,*de,*dm;
  check(cudaMalloc(&dh,count*4));check(cudaMalloc(&di,count*4));check(cudaMalloc(&ds,cells*4));check(cudaMalloc(&de,cells*4));check(cudaMalloc(&dm,map.size()*4));
  check(cudaMemcpy(dh,hashes.data(),count*4,cudaMemcpyHostToDevice));check(cudaMemcpy(di,indices.data(),count*4,cudaMemcpyHostToDevice));check(cudaMemcpy(dm,map.data(),map.size()*4,cudaMemcpyHostToDevice));check(cudaMemset(ds,0,cells*4));check(cudaMemset(de,0,cells*4));
  unsigned blocks=n?(n+127)/128:1;
  findCellStartEndD<<<blocks,128,129*4>>>(ds,de,dh,di,n);
  OriginalToSortedD<<<blocks,128>>>(dm,di,n);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  check(cudaMemcpy(start.data(),ds,cells*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(end.data(),de,cells*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(map.data(),dm,map.size()*4,cudaMemcpyDeviceToHost));
  printf("%s{\"n\":%u,\"start\":",first?"":",",n);array(start);printf(",\"end\":");array(end);printf(",\"map\":");array(map);printf("}");first=false;
  for(auto p:{dh,di,ds,de,dm})check(cudaFree(p));
 }
 printf("]}\n");return 0;
}
