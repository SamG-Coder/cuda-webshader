// MIT reference harness around unchanged Chrono kernel/helper bodies.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include "chrono-shifting.cu"
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>T* upload(std::ifstream& f,size_t n){std::vector<T> h(n);f.read((char*)h.data(),n*sizeof(T));if(!f)exit(2);T* d;check(cudaMalloc(&d,n*sizeof(T)));check(cudaMemcpy(d,h.data(),n*sizeof(T),cudaMemcpyHostToDevice));return d;}
template<class T>void dump(std::ofstream& f,T* p,size_t n){std::vector<T> h(n);check(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));f.write((char*)h.data(),n*sizeof(T));}
int main(){const uint n=30327,total=794643;ChFsiParamsSPH p;std::ifstream params(".local/chrono-params.bin",std::ios::binary);params.read((char*)&p,sizeof(p));if(!params)return 2;check(cudaMemcpyToSymbol(paramsD,&p,sizeof(p)));
 std::ifstream input("reports/chrono-adami-input.bin",std::ios::binary);auto offsets=upload<uint>(input,n+1),neighbors=upload<uint>(input,total);auto pos=upload<Real4>(input,n),rho=upload<Real4>(input,n);auto vel=upload<Real3>(input,n);Real3* acc;bool* flag;check(cudaMalloc(&acc,n*sizeof(Real3)));check(cudaMemset(acc,0,n*sizeof(Real3)));check(cudaMalloc(&flag,4));check(cudaMemset(flag,0,4));
 CfdAdamiBC_D<<<(n+127)/128,128>>>(offsets,neighbors,pos,n,acc,rho,vel,flag);check(cudaGetLastError());check(cudaDeviceSynchronize());
 std::ofstream state(".local/chrono-rhs-input-properties.bin",std::ios::binary);dump(state,rho,n);dump(state,vel,n);state.close();
 Real4* derivatives;uint* surface;Real *divergence,*courant,*acceleration;
 check(cudaMalloc(&derivatives,n*16));check(cudaMalloc(&surface,n*4));check(cudaMalloc(&divergence,n*4));check(cudaMalloc(&courant,n*4));check(cudaMalloc(&acceleration,n*4));
 check(cudaMemset(derivatives,0,n*16));check(cudaMemset(surface,0,n*4));check(cudaMemset(divergence,0,n*4));check(cudaMemset(courant,0,n*4));check(cudaMemset(acceleration,0,n*4));
 CfdCalcRHS_D<<<(n+127)/128,128>>>(derivatives,pos,vel,rho,offsets,neighbors,n,surface,divergence,courant,acceleration,flag);check(cudaGetLastError());check(cudaDeviceSynchronize());
 Real3* shifting;check(cudaMalloc(&shifting,n*12));check(cudaMemset(shifting,0,n*12));
 if(p.shifting_method!=ShiftingMethod::XSPH)return 4;
 Calc_Shifting_D<ShiftingMethod::XSPH><<<(n+127)/128,128>>>(shifting,pos,vel,rho,offsets,neighbors,n,divergence,flag);check(cudaGetLastError());check(cudaDeviceSynchronize());
 std::ofstream output("reports/chrono-shifting-native.bin",std::ios::binary);dump(output,shifting,n);dump(output,(unsigned char*)flag,4);
 std::vector<Real3> moving(n);
 for(uint i=0;i<n;i++)moving[i]=make_Real3((int(i%17)-8)*0.125f,(int(i%13)-6)*0.0625f,(int(i%11)-5)*0.25f);
 check(cudaMemcpy(vel,moving.data(),n*12,cudaMemcpyHostToDevice));
 Calc_Shifting_D<ShiftingMethod::XSPH><<<(n+127)/128,128>>>(shifting,pos,vel,rho,offsets,neighbors,n,divergence,flag);check(cudaGetLastError());check(cudaDeviceSynchronize());
 dump(output,shifting,n);dump(output,(unsigned char*)flag,4);
 check(cudaFree(shifting));
 for(void* ptr:{(void*)derivatives,(void*)surface,(void*)divergence,(void*)courant,(void*)acceleration})check(cudaFree(ptr));

 for(void* ptr:{(void*)offsets,(void*)neighbors,(void*)pos,(void*)rho,(void*)vel,(void*)acc,(void*)flag})check(cudaFree(ptr));return output?0:3;
}
