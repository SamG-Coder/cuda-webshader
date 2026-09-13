// MIT native reference: retain the faulty upstream baseline and validate explicit normalization.
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "chrono-activity-scan.cuh"
#include "chrono-compact.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
#include "chrono-selected-native.cuh"
template<class T>void write(std::ofstream& f,const std::vector<T>& v){f.write((const char*)v.data(),v.size()*sizeof(T));}
int main(){
 std::ifstream activity("reports/chrono-activity-native.bin",std::ios::binary),inputs("reports/chrono-activity-input.bin",std::ios::binary);
 std::ofstream output("reports/chrono-compact-native.bin",std::ios::binary),json("reports/chrono-compact-native.json");
 size_t activityOffset=0,inputOffset=0;json<<"[\n";
 for(int mode=0;mode<14;mode++){
  int n=mode?198:30327;std::vector<int> flags(n),baseline(n),prefix(n);std::vector<float4> positions(n);
  activity.seekg(activityOffset+n*4);activity.read((char*)flags.data(),n*4);activityOffset+=n*20;
  inputs.seekg(inputOffset);inputs.read((char*)positions.data(),n*16);inputOffset+=n*44+312;if(!activity||!inputs)return 2;
  thrust::device_vector<int> original(flags),rawPrefix(n),normalized(n),fixedPrefix(n);thrust::device_vector<uint> list(n,0xffffffffu);
  thrust::exclusive_scan(thrust::device,original.begin(),original.end(),rawPrefix.begin(),0,ActivityScanOp());
  thrust::copy(rawPrefix.begin(),rawPrefix.end(),baseline.begin());
  normalizeActivity<<<(n+127)/128,128>>>(thrust::raw_pointer_cast(original.data()),(uint*)thrust::raw_pointer_cast(normalized.data()),n);check(cudaGetLastError());
  // Original functor is now evaluated within its associative, nonnegative domain.
  thrust::exclusive_scan(thrust::device,normalized.begin(),normalized.end(),fixedPrefix.begin(),0,ActivityScanOp());
  thrust::copy(fixedPrefix.begin(),fixedPrefix.end(),prefix.begin());
  int expected=0,mismatches=0,collisions=0;std::vector<int> owners(n,-1);
  for(int i=0;i<n;i++){mismatches+=baseline[i]!=expected;if(prefix[i]!=expected)return 3;if(flags[i]>0){expected++;if(baseline[i]>=0&&baseline[i]<n){collisions+=owners[baseline[i]]!=-1;owners[baseline[i]]=i;}}}
  fillActiveListD<<<(n+127)/128,128>>>((uint*)thrust::raw_pointer_cast(fixedPrefix.data()),thrust::raw_pointer_cast(original.data()),thrust::raw_pointer_cast(list.data()),n);check(cudaGetLastError());
  const int total=prefix.back()+(flags.back()>0);std::vector<uint> compact(n);thrust::copy(list.begin(),list.end(),compact.begin());
  thrust::device_vector<float4> pos(positions),selected(total?total:1);if(total){gatherSelected<<<(total+127)/128,128>>>(thrust::raw_pointer_cast(pos.data()),thrust::raw_pointer_cast(list.data()),thrust::raw_pointer_cast(selected.data()),total);check(cudaGetLastError());}check(cudaDeviceSynchronize());
  std::vector<float4> gathered(total);thrust::copy(selected.begin(),selected.begin()+total,gathered.begin());
  int at=0;for(int i=0;i<n;i++)if(flags[i]>0&&compact[at++]!=(uint)i)return 4;
  selectedNeighbors(mode,pos,list,total);
  const auto offset=(long long)output.tellp();write(output,baseline);write(output,prefix);write(output,compact);write(output,gathered);
  json<<(mode?",\n":"")<<"{\"mode\":"<<mode<<",\"n\":"<<n<<",\"total\":"<<total<<",\"offset\":"<<offset<<",\"baselinePrefixMismatches\":"<<mismatches<<",\"baselineWriteCollisions\":"<<collisions<<",\"baselineTotal\":"<<baseline.back()+(flags.back()>0)<<"}";
 }
 json<<"\n]\n";if(!output||!json)return 5;printf("Native activity scan/compaction: baseline captured, normalized prefixes and marker IDs verified\n");
}
