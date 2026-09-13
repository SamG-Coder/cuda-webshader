// MIT native data fixture for the unchanged Chrono reorder kernel and its diagnostics.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <vector>
#include "chrono-reorder.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>T* upload(const std::vector<T>& v){T* p;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T>void write(std::ofstream& f,const std::vector<T>& v){f.write((const char*)v.data(),v.size()*sizeof(T));}
template<class T>void dump(std::ofstream& f,T* p,unsigned n){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));write(f,v);}
int main(){
 const unsigned m=521,n=259;ChFsiParamsSPH p;std::ifstream params(".local/chrono-params.bin",std::ios::binary);params.read((char*)&p,sizeof(p));if(!params)return 2;
 std::ofstream inputs("reports/chrono-reorder-input.bin",std::ios::binary),outputs("reports/chrono-reorder-native.bin",std::ios::binary),json("reports/chrono-reorder-native.json");json<<"[\n";
 for(int mode=0;mode<4;mode++){
  std::vector<uint> indices(n);for(unsigned i=0;i<n;i++)indices[i]=(2*i+3)%m;
  std::vector<Real4> pos(m),rho(m);std::vector<Real3> vel(m),xx(m),xy(m),pc(m);std::vector<int32_t> activity(m);
  for(unsigned i=0;i<m;i++){float f=float(i);pos[i]={f*.125f,-f*.25f,f*.5f,.1f};rho[i]={1000+f*.25f,f*.5f,.001f,float(int(i%6)-3)};vel[i]={f+1,-f-2,f+3};xx[i]={f+4,f+5,f+6};xy[i]={f+7,f+8,f+9};pc[i]={f+10,f+11,f+12};activity[i]=int(i%3)-1;}
  if(mode>=2){pos[indices[6]].x=std::numeric_limits<float>::infinity();xx[indices[129]].x=std::numeric_limits<float>::quiet_NaN();}
  auto inputOffset=(long long)inputs.tellp(),outputOffset=(long long)outputs.tellp();write(inputs,indices);write(inputs,pos);write(inputs,vel);write(inputs,rho);write(inputs,xx);write(inputs,xy);write(inputs,pc);write(inputs,activity);
  auto ids=upload(indices);auto ap=upload(activity);auto pp=upload(pos),rp=upload(rho);auto vp=upload(vel),xp=upload(xx),yp=upload(xy),cp=upload(pc);
  auto so=upload(std::vector<Real4>(n,{42,42,42,42})),sr=upload(std::vector<Real4>(n,{42,42,42,42}));auto sv=upload(std::vector<Real3>(n,{42,42,42})),sx=upload(std::vector<Real3>(n,{42,42,42})),sy=upload(std::vector<Real3>(n,{42,42,42})),sc=upload(std::vector<Real3>(n,{42,42,42}));auto sa=upload(std::vector<int32_t>(n,42));
  p.physics_problem=mode%2?PhysicsProblem::CRM:PhysicsProblem::CFD;check(cudaMemcpyToSymbol(paramsD,&p,sizeof(p)));printf("CASE %d\n",mode);fflush(stdout);
  reorderDataD<<<(n+127)/128,128>>>(ids,so,sv,sr,sx,sy,sc,sa,pp,vp,rp,xp,yp,cp,ap,n);check(cudaGetLastError());check(cudaDeviceSynchronize());
  dump(outputs,so,n);dump(outputs,sv,n);dump(outputs,sr,n);dump(outputs,sx,n);dump(outputs,sy,n);dump(outputs,sc,n);dump(outputs,sa,n);
  json<<(mode?",\n":"")<<"{\"mode\":"<<mode<<",\"original\":"<<m<<",\"selected\":"<<n<<",\"inputOffset\":"<<inputOffset<<",\"outputOffset\":"<<outputOffset<<"}";
  for(void* b:std::vector<void*>{ids,ap,pp,rp,vp,xp,yp,cp,so,sr,sv,sx,sy,sc,sa})check(cudaFree(b));
 }
 json<<"\n]\n";return inputs&&outputs&&json?0:3;
}
