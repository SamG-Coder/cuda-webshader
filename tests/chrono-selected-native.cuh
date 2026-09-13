// MIT native chain validation, called with GPU-produced compact indices.
#include <thrust/sort.h>
#include "chrono-search.cu"
static_assert(sizeof(Real4)==sizeof(float4),"Position buffers must retain four floats");
template<class T>T* selectedPtr(thrust::device_vector<T>& v){return thrust::raw_pointer_cast(v.data());}
template<class T>void selectedDump(std::ofstream& f,thrust::device_vector<T>& v){
 if(v.empty())return;std::vector<T> host(v.size());check(cudaMemcpy(host.data(),selectedPtr(v),host.size()*sizeof(T),cudaMemcpyDeviceToHost));f.write((const char*)host.data(),host.size()*sizeof(T));
}
static void selectedNeighbors(int mode,thrust::device_vector<float4>& original,thrust::device_vector<uint>& active,unsigned n){
 static std::ofstream binary("reports/chrono-selected-native.bin",std::ios::binary),json("reports/chrono-selected-native.json");
 ChFsiParamsSPH p;std::ifstream file(".local/chrono-params.bin",std::ios::binary);file.read((char*)&p,sizeof(p));if(!file)exit(6);
 if(mode){p.worldOrigin={-1,-1,-1};p.boxDims={2,2,2};p.gridSize=make_int3(10,10,10);p.cellSize={.2f,.2f,.2f};const int flags=mode>=5&&mode<=12?mode-5:0;p.x_periodic=flags&1;p.y_periodic=flags&2;p.z_periodic=flags&4;}
 check(cudaMemcpyToSymbol(paramsD,&p,sizeof(p)));const unsigned cells=p.gridSize.x*p.gridSize.y*p.gridSize.z,blocks=(n+127)/128;
 thrust::device_vector<Real4> positions(original.size()),sorted(n),unused(1);
 check(cudaMemcpy(selectedPtr(positions),selectedPtr(original),original.size()*sizeof(Real4),cudaMemcpyDeviceToDevice));
 thrust::device_vector<uint> hashes(n),indices(n),start(cells),end(cells),counts(n+1),offsets(n+1);
 if(n){
  hashSelected<<<blocks,128>>>(selectedPtr(positions),selectedPtr(active),selectedPtr(hashes),selectedPtr(indices),n);check(cudaGetLastError());
  thrust::sort_by_key(hashes.begin(),hashes.end(),indices.begin());
  gatherMarkers<<<blocks,128>>>(selectedPtr(positions),selectedPtr(indices),selectedPtr(sorted),n);
  findCellStartEndD<<<blocks,128,129*4>>>(selectedPtr(start),selectedPtr(end),selectedPtr(hashes),selectedPtr(indices),n);
  neighborSearchNum<<<blocks,128>>>(selectedPtr(sorted),selectedPtr(unused),selectedPtr(start),selectedPtr(end),n,selectedPtr(counts));check(cudaGetLastError());
 }
 thrust::exclusive_scan(counts.begin(),counts.end(),offsets.begin());const uint total=offsets[n];thrust::device_vector<uint> neighbors(total);
 if(n){neighborSearchID<<<blocks,128>>>(selectedPtr(sorted),selectedPtr(unused),selectedPtr(start),selectedPtr(end),n,selectedPtr(offsets),selectedPtr(neighbors));check(cudaGetLastError());}check(cudaDeviceSynchronize());
 const auto offset=(long long)binary.tellp();selectedDump(binary,hashes);selectedDump(binary,indices);selectedDump(binary,start);selectedDump(binary,end);selectedDump(binary,counts);selectedDump(binary,offsets);selectedDump(binary,neighbors);selectedDump(binary,sorted);
 json<<(mode?",\n":"[\n")<<"{\"mode\":"<<mode<<",\"n\":"<<n<<",\"cells\":"<<cells<<",\"neighbors\":"<<total<<",\"offset\":"<<offset<<",\"sections\":["<<n<<","<<n<<","<<cells<<","<<cells<<","<<n+1<<","<<n+1<<","<<total<<","<<4*n<<"]}";
 if(mode==13)json<<"\n]\n";if(!binary||!json)exit(7);
}
