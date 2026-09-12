// MIT validation harness. Read the actual initialized Chrono parameter capture.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include "chrono-hash.cu"
static_assert(std::is_same<decltype(floor(1.f)),float>::value,"CUDA floor float overload");
static_assert(sizeof(ChFsiParamsSPH)==608,"Pinned Chrono parameter ABI changed");
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){
 FILE* f=fopen(".local/chrono-params.bin","rb");if(!f)return 1;ChFsiParamsSPH params;if(fread(&params,sizeof(params),1,f)!=1||fgetc(f)!=EOF)return 2;fclose(f);
 f=fopen("reports/chrono-hash-input.bin","rb");if(!f)return 3;fseek(f,0,SEEK_END);long bytes=ftell(f);rewind(f);if(bytes%12)return 4;unsigned n=bytes/12;std::vector<Real3> points(n);if(fread(points.data(),12,n,f)!=n)return 5;fclose(f);
 Real3* dp;unsigned* dh;int* db;check(cudaMalloc(&dp,bytes));check(cudaMalloc(&dh,n*4));check(cudaMalloc(&db,n*12));check(cudaMemcpy(dp,points.data(),bytes,cudaMemcpyHostToDevice));
 f=fopen("reports/chrono-hash-native.bin","wb");if(!f)return 6;
 std::vector<unsigned> hashes(n);std::vector<int> bins(n*3);
 // First case uses the actual dam-break flags; then exercise all combinations.
 for(int mode=-1;mode<8;mode++){
  auto p=params;if(mode>=0){p.x_periodic=bool(mode&1);p.y_periodic=bool(mode&2);p.z_periodic=bool(mode&4);}
  check(cudaMemcpyToSymbol(paramsD,&p,sizeof(p)));hashProbe<<<(n+127)/128,128>>>(dp,dh,db,n);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(hashes.data(),dh,n*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(bins.data(),db,n*12,cudaMemcpyDeviceToHost));
  if(fwrite(hashes.data(),4,n,f)!=n||fwrite(bins.data(),4,n*3,f)!=n*3)return 7;
 }
 fclose(f);check(cudaFree(dp));check(cudaFree(dh));check(cudaFree(db));printf("%u positions, 9 boundary configurations, %u integer values captured\n",n,n*4*9);return 0;
}
