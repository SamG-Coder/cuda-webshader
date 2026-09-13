// MIT native chain validation, called with GPU-produced compact indices.
#include <thrust/sort.h>
#include "chrono-search.cu"
#include "chrono-selected-reorder.cuh"
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
 const unsigned m=original.size();if(p.physics_problem!=PhysicsProblem::CFD)exit(8);
 const size_t inOffset=mode?30327*44+312+(mode-1)*(198*44+312):0,outOffset=mode?30327*20+(mode-1)*198*20:0;
 std::vector<Real4> hostRho(m);std::vector<Real3> hostVel(m);std::vector<int32_t> hostActivity(m);
 std::ifstream input("reports/chrono-activity-input.bin",std::ios::binary),output("reports/chrono-activity-native.bin",std::ios::binary);
 input.seekg(inOffset+16*m);input.read((char*)hostRho.data(),16*m);
 output.seekg(outOffset);output.read((char*)hostActivity.data(),4*m);output.seekg(outOffset+8*m);output.read((char*)hostVel.data(),12*m);if(!input||!output)exit(9);
 thrust::device_vector<Real4> positions(m),sorted(n),rho(hostRho),sortedRho(n);
 thrust::device_vector<Real3> vel(hostVel),sortedVel(n),unused(1),stress1(1),stress2(1),stress3(1);
 thrust::device_vector<int32_t> activity(hostActivity),sortedActivity(n);
 thrust::device_vector<uint> map(m,UINT_MAX);
 check(cudaMemcpy(selectedPtr(positions),selectedPtr(original),original.size()*sizeof(Real4),cudaMemcpyDeviceToDevice));
 thrust::device_vector<uint> hashes(n),indices(n),start(cells),end(cells),counts(n+1),offsets(n+1);
 if(n){
  hashSelected<<<blocks,128>>>(selectedPtr(positions),selectedPtr(active),selectedPtr(hashes),selectedPtr(indices),n);check(cudaGetLastError());
  thrust::sort_by_key(hashes.begin(),hashes.end(),indices.begin());
  OriginalToSortedD<<<blocks,128>>>(selectedPtr(map),selectedPtr(indices),n);
  reorderDataD<<<blocks,128>>>(selectedPtr(indices),selectedPtr(sorted),selectedPtr(sortedVel),selectedPtr(sortedRho),selectedPtr(stress1),selectedPtr(stress2),selectedPtr(stress3),selectedPtr(sortedActivity),selectedPtr(positions),selectedPtr(vel),selectedPtr(rho),selectedPtr(unused),selectedPtr(unused),selectedPtr(unused),selectedPtr(activity),n);check(cudaGetLastError());
  findCellStartEndD<<<blocks,128,129*4>>>(selectedPtr(start),selectedPtr(end),selectedPtr(hashes),selectedPtr(indices),n);
  neighborSearchNum<<<blocks,128>>>(selectedPtr(sorted),selectedPtr(sortedRho),selectedPtr(start),selectedPtr(end),n,selectedPtr(counts));check(cudaGetLastError());
 }
 thrust::exclusive_scan(counts.begin(),counts.end(),offsets.begin());const uint total=offsets[n];thrust::device_vector<uint> neighbors(total);
 if(n){neighborSearchID<<<blocks,128>>>(selectedPtr(sorted),selectedPtr(sortedRho),selectedPtr(start),selectedPtr(end),n,selectedPtr(offsets),selectedPtr(neighbors));check(cudaGetLastError());}check(cudaDeviceSynchronize());
 const auto offset=(long long)binary.tellp();selectedDump(binary,hashes);selectedDump(binary,indices);selectedDump(binary,start);selectedDump(binary,end);selectedDump(binary,counts);selectedDump(binary,offsets);selectedDump(binary,neighbors);selectedDump(binary,sorted);selectedDump(binary,map);selectedDump(binary,sortedVel);selectedDump(binary,sortedRho);selectedDump(binary,sortedActivity);
 json<<(mode?",\n":"[\n")<<"{\"mode\":"<<mode<<",\"n\":"<<n<<",\"cells\":"<<cells<<",\"neighbors\":"<<total<<",\"offset\":"<<offset<<",\"sections\":["<<n<<","<<n<<","<<cells<<","<<cells<<","<<n+1<<","<<n+1<<","<<total<<","<<4*n<<","<<m<<","<<3*n<<","<<4*n<<","<<n<<"]}";
 if(mode==13)json<<"\n]\n";if(!binary||!json)exit(7);
}
