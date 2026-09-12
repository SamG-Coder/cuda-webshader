// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <vector>
#include <cstdio>
int main(){
 const int n=257;
 for(int frame:{1,8,32,64}){
  char path[160];snprintf(path,sizeof(path),"reports/smoke-integration-%d-depth.bin",frame);
  std::vector<float> keys(n);FILE*f=fopen(path,"rb");if(!f)return 2;size_t got=fread(keys.data(),4,n,f);fclose(f);if(got!=n)return 2;
  thrust::device_vector<float> dk(keys);thrust::device_vector<unsigned> dv(n);thrust::sequence(dv.begin(),dv.end());thrust::sort_by_key(dk.begin(),dk.end(),dv.begin());
  std::vector<unsigned> values(n);thrust::copy(dk.begin(),dk.end(),keys.begin());thrust::copy(dv.begin(),dv.end(),values.begin());
  snprintf(path,sizeof(path),"reports/smoke-sort-%d-native.bin",frame);f=fopen(path,"wb");if(!f)return 2;bool ok=fwrite(keys.data(),4,n,f)==n&&fwrite(values.data(),4,n,f)==n;fclose(f);if(!ok)return 2;
  printf("Captured native Thrust float depth sort: step %d, %d particles.\n",frame,n);
 }
 return cudaDeviceSynchronize()==cudaSuccess?0:2;
}
