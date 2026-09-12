// MIT native reference driver for Chrono's unchanged neighbour-search kernels.
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include "chrono-search.cu"
static_assert(std::is_same<decltype(rint(1.f)),float>::value,"CUDA rint float overload");
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T> T* ptr(thrust::device_vector<T>& v){return thrust::raw_pointer_cast(v.data());}
template<class T> void dump(FILE* file,thrust::device_vector<T>& v){std::vector<T> data(v.size());check(cudaMemcpy(data.data(),ptr(v),data.size()*sizeof(T),cudaMemcpyDeviceToHost));if(fwrite(data.data(),sizeof(T),data.size(),file)!=data.size())exit(2);}
int main(){
 ChFsiParamsSPH params;FILE* file=fopen(".local/chrono-params.bin","rb");if(!file||fread(&params,sizeof(params),1,file)!=1||fgetc(file)!=EOF)return 3;fclose(file);check(cudaMemcpyToSymbol(paramsD,&params,sizeof(params)));
 file=fopen("reports/chrono-search-input.bin","rb");if(!file)return 4;fseek(file,0,SEEK_END);long bytes=ftell(file);rewind(file);if(bytes%sizeof(Real4))return 5;unsigned n=bytes/sizeof(Real4);std::vector<Real4> input(n);if(fread(input.data(),sizeof(Real4),n,file)!=n)return 6;fclose(file);
 unsigned cells=params.gridSize.x*params.gridSize.y*params.gridSize.z,blocks=(n+127)/128;
 thrust::device_vector<Real4> positions(input),sorted(n),unused(1);
 thrust::device_vector<unsigned> hashes(n),indices(n),start(cells),end(cells),counts(n+1),offsets(n+1);
 hashMarkers<<<blocks,128>>>(ptr(positions),ptr(hashes),ptr(indices),n);check(cudaGetLastError());
 thrust::sort_by_key(hashes.begin(),hashes.end(),indices.begin());
 gatherMarkers<<<blocks,128>>>(ptr(positions),ptr(indices),ptr(sorted),n);
 findCellStartEndD<<<blocks,128,129*4>>>(ptr(start),ptr(end),ptr(hashes),ptr(indices),n);
 neighborSearchNum<<<blocks,128>>>(ptr(sorted),ptr(unused),ptr(start),ptr(end),n,ptr(counts));check(cudaGetLastError());check(cudaDeviceSynchronize());
 thrust::exclusive_scan(counts.begin(),counts.end(),offsets.begin());unsigned total=offsets[n];
 thrust::device_vector<unsigned> neighbors(total);
 neighborSearchID<<<blocks,128>>>(ptr(sorted),ptr(unused),ptr(start),ptr(end),n,ptr(offsets),ptr(neighbors));check(cudaGetLastError());check(cudaDeviceSynchronize());
 file=fopen("reports/chrono-search-native.bin","wb");if(!file)return 7;
 dump(file,hashes);dump(file,indices);dump(file,start);dump(file,end);dump(file,counts);dump(file,offsets);dump(file,neighbors);fclose(file);
 file=fopen("reports/chrono-search-native.json","w");if(!file)return 8;fprintf(file,"{\"markers\":%u,\"fluid\":16731,\"cells\":%u,\"neighbors\":%u,\"sections\":[%u,%u,%u,%u,%u,%u,%u]}",n,cells,total,n,n,cells,cells,n+1,n+1,total);fclose(file);
 printf("%u markers, %u neighbor entries\n",n,total);return 0;
}
