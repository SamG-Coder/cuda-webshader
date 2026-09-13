// MIT validation harness for the unchanged Chrono UpdateActivityD kernel.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include "chrono-activity.cu"
#include "native-records.cu"
static_assert(sizeof(Flags)==24 && offsetof(Flags,p)==4 && offsetof(Flags,c)==16 && offsetof(Flags,w)==20,"Native adjacent bool/float3 layout");
static_assert(sizeof(ActiveDomain)==52 && offsetof(ActiveDomain,a_min)==4 && offsetof(ActiveDomain,e_max)==40,"Native domain layout");
static_assert(sizeof(Counters)==176 && sizeof(ChFsiParamsSPH)==608,"Native constant layouts");
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>T* upload(const std::vector<T>& v){T* p;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T>void write(std::ofstream& f,const std::vector<T>& v){f.write((const char*)v.data(),v.size()*sizeof(T));}
int main(){
 ChFsiParamsSPH actual{};std::ifstream params(".local/chrono-params.bin",std::ios::binary);params.read((char*)&actual,sizeof(actual));if(!params)return 2;
 std::ofstream inputs("reports/chrono-activity-input.bin",std::ios::binary),outputs("reports/chrono-activity-native.bin",std::ios::binary),json("reports/chrono-activity-native.json");json<<"[\n";
 const float coords[]={-2,-1,-.75f,-.5f,-.25f,0,.25f,.5f,.75f,1,2};
 for(int mode=0;mode<14;mode++){
  const int n=mode==0?30327:198;ChFsiParamsSPH p=actual;Counters counts{};counts.numAllMarkers=n;counts.numFluidMarkers=mode==0?16731:99;
  std::vector<Real4> positions(n),rho(n);std::vector<Real3> velocities(n);
  if(mode==0){std::ifstream f("reports/chrono-search-input.bin",std::ios::binary);f.read((char*)positions.data(),n*sizeof(Real4));if(!f)return 3;std::ifstream r("reports/chrono-marker-rhopremu.bin",std::ios::binary),v("reports/chrono-marker-velocities.bin",std::ios::binary);r.read((char*)rho.data(),n*sizeof(Real4));v.read((char*)velocities.data(),n*sizeof(Real3));if(!r||!v)return 3;}
  else{p.worldOrigin={-1,-1,-1};p.boxDims={2,2,2};p.free_flow_duration=.5f;p.x_periodic=p.y_periodic=p.z_periodic=false;
   for(int i=0;i<n;i++){const int axis=(i/11)%3,kind=i/33;Real3 xyz{0,0,0};if(axis==0)xyz.x=coords[i%11];if(axis==1)xyz.y=coords[i%11];if(axis==2)xyz.z=coords[i%11];positions[i]={xyz.x,xyz.y,xyz.z,.1f};rho[i]={1000,0,.001f,float(kind-3)};velocities[i]={float(i+1),float(-i-1),.25f};}
  }
  std::vector<ActiveDomain> domains(6);memset(domains.data(),0xa5,domains.size()*sizeof(ActiveDomain));
  for(int i=0;i<6;i++){domains[i].inverted=(i%2)==0;domains[i].a_min={-10,-10,-10};domains[i].a_max={10,10,10};domains[i].e_min={-10,-10,-10};domains[i].e_max={10,10,10};}
  domains[1].a_min={-.25f,-.25f,-.25f};domains[1].a_max={.25f,.25f,.25f};domains[1].e_min={-.5f,-.5f,-.5f};domains[1].e_max={.5f,.5f,.5f};
  domains[3].a_min={.5f,-1,-1};domains[3].a_max={.75f,1,1};domains[3].e_min={.25f,-1,-1};domains[3].e_max={1,1,1};
  domains[5].a_min={-1,-1,-.75f};domains[5].a_max={1,1,-.5f};domains[5].e_min={-1,-1,-1};domains[5].e_max={1,1,-.25f};
  const bool has_ad=mode>=2;double time=mode==2?.5-0x1p-40:mode==4?.5+0x1p-40:.5;
  if(mode>=5&&mode<=12){const int flags=mode-5;p.x_periodic=flags&1;p.y_periodic=flags&2;p.z_periodic=flags&4;}
  if(has_ad&&mode!=13)counts.numFsiBodies=counts.numFsiNodes1D=counts.numFsiNodes2D=2;
  const auto inputOffset=(long long)inputs.tellp(),outputOffset=(long long)outputs.tellp();write(inputs,positions);write(inputs,rho);write(inputs,velocities);write(inputs,domains);
  auto pos=upload(positions),prop=upload(rho);auto vel=upload(velocities),unused=upload(std::vector<Real3>(1));auto domain=upload(domains);auto active=upload(std::vector<int32_t>(n,77)),extended=upload(std::vector<int32_t>(n,77));
  check(cudaMemcpyToSymbol(paramsD,&p,sizeof(p)));check(cudaMemcpyToSymbol(countersD,&counts,sizeof(counts)));
  UpdateActivityD<<<(n+127)/128,128>>>(pos,vel,unused,unused,unused,has_ad,domain,domain+2,domain+4,active,extended,prop,time);check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<int32_t> a(n),e(n);check(cudaMemcpy(a.data(),active,n*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(e.data(),extended,n*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(velocities.data(),vel,n*12,cudaMemcpyDeviceToHost));write(outputs,a);write(outputs,e);write(outputs,velocities);
  unsigned int bits[2];memcpy(bits,&time,8);
  json<<(mode?",\n":"")<<"{\"mode\":"<<mode<<",\"n\":"<<n<<",\"inputOffset\":"<<inputOffset<<",\"outputOffset\":"<<outputOffset<<",\"has_ad\":"<<has_ad<<",\"time.lo\":"<<bits[0]<<",\"time.hi\":"<<bits[1]<<",\"domainCount\":"<<counts.numFsiBodies<<",\"periodic\":["<<p.x_periodic<<","<<p.y_periodic<<","<<p.z_periodic<<"]}";
  check(cudaFree(pos));check(cudaFree(prop));check(cudaFree(vel));check(cudaFree(unused));check(cudaFree(domain));check(cudaFree(active));check(cudaFree(extended));
 }
 std::vector<Flags> values(2);memset(values.data(),0xa5,48);for(int i=0;i<2;i++){values[i].a=i;values[i].b=!i;values[i].c=i;values[i].p=make_float3(1+i*3,2+i*3,3+i*3);values[i].w=i?11:7;}
 std::ofstream recordInput("reports/native-records-input.bin",std::ios::binary);write(recordInput,values);
 auto d=upload(values);auto o=upload(std::vector<float>(4));auto flags=upload(std::vector<int>{0,1,2,3,4});nativeRecords<<<1,1>>>(d,o,flags);check(cudaGetLastError());check(cudaDeviceSynchronize());
 float out[4];int fl[5];check(cudaMemcpy(out,o,sizeof(out),cudaMemcpyDeviceToHost));check(cudaMemcpy(fl,flags,sizeof(fl),cudaMemcpyDeviceToHost));
 std::ofstream recordOutput("reports/native-records-native.json");recordOutput<<"[";for(int i=0;i<4;i++)recordOutput<<(i?",":"")<<out[i];for(int i=0;i<5;i++)recordOutput<<","<<fl[i];recordOutput<<"]\n";if(!recordInput||!recordOutput)return 5;
 check(cudaFree(d));check(cudaFree(o));check(cudaFree(flags));
 json<<"\n]\n";if(!inputs||!outputs||!json)return 4;printf("Original UpdateActivityD: 30327 initialized marker positions and 13 boundary/activity cases\n");
}
