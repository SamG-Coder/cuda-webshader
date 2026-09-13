// MIT reference harness around unchanged Chrono kernel/helper bodies.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include "chrono-rk2.cu"
#include "chrono-scatter.cuh"
#include "chrono-loop-reorder.cuh"
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>T* upload(std::ifstream& f,size_t n){std::vector<T> h(n);f.read((char*)h.data(),n*sizeof(T));if(!f)exit(2);T* d;check(cudaMalloc(&d,n*sizeof(T)));check(cudaMemcpy(d,h.data(),n*sizeof(T),cudaMemcpyHostToDevice));return d;}
template<class T>void dump(std::ofstream& f,T* p,size_t n){std::vector<T> h(n);check(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));f.write((char*)h.data(),n*sizeof(T));}

// Diagnostic native replay: all 30327 markers remain selected in the bounded
// runs compared here. The production GPU path recomputes activity each step.
int main(int argc,char**argv){const uint n=30327;const int steps=argc>1?atoi(argv[1]):100;if(steps<1||steps>100000)return 4;ChFsiParamsSPH p;std::ifstream pf(".local/chrono-params.bin",std::ios::binary);pf.read((char*)&p,sizeof(p));if(!pf)return 2;if(p.physics_problem!=PhysicsProblem::CFD||p.shifting_method!=ShiftingMethod::XSPH||p.num_proximity_search_steps!=1)return 5;check(cudaMemcpyToSymbol(paramsD,&p,sizeof(p)));
 std::ifstream ip("reports/chrono-search-input.bin",std::ios::binary),iv("reports/chrono-marker-velocities.bin",std::ios::binary),ir("reports/chrono-marker-rhopremu.bin",std::ios::binary);
 auto originalPos=upload<Real4>(ip,n),originalRho=upload<Real4>(ir,n);auto originalVel=upload<Real3>(iv,n);
 auto alloc=[](auto**ptr,size_t bytes){check(cudaMalloc(ptr,bytes));check(cudaMemset(*ptr,0,bytes));};
 Real4 *pos,*rho,*tmpPos,*tmpRho,*deriv,*outDeriv;Real3 *vel,*tmpVel,*acc,*shifting,*tauA,*tauB,*pc,*dtauA,*dtauB,*outA,*outB,*outPc;uint *hashes,*indices,*active,*start,*end,*counts,*offsets,*neighbors,*surface;int32_t *activity,*sortedActivity;Real *divergence,*courant,*acceleration;bool*flag;
 for(auto ptr:{&pos,&rho,&tmpPos,&tmpRho,&deriv,&outDeriv})alloc(ptr,n*16);
 for(auto ptr:{&vel,&tmpVel,&acc,&shifting,&tauA,&tauB,&pc,&dtauA,&dtauB,&outA,&outB,&outPc})alloc(ptr,n*12);
 for(auto ptr:{&hashes,&indices,&active,&surface})alloc(ptr,n*4);
 alloc(&activity,n*4);alloc(&sortedActivity,n*4);alloc(&flag,4);
 for(auto ptr:{&divergence,&courant,&acceleration})alloc(ptr,n*4);
 const uint cells=p.gridSize.x*p.gridSize.y*p.gridSize.z;alloc(&start,cells*4);alloc(&end,cells*4);alloc(&counts,(n+1)*4);alloc(&offsets,(n+1)*4);
 std::vector<uint> ids(n);std::vector<int32_t> ones(n,1);for(uint i=0;i<n;i++)ids[i]=i;check(cudaMemcpy(active,ids.data(),n*4,cudaMemcpyHostToDevice));check(cudaMemcpy(activity,ones.data(),n*4,cudaMemcpyHostToDevice));
 const uint blocks=(n+127)/128;
 for(int step=0;step<steps;step++){
  check(cudaMemset(start,0,cells*4));check(cudaMemset(end,0,cells*4));
  hashSelected<<<blocks,128>>>(originalPos,active,hashes,indices,n);thrust::sort_by_key(thrust::device_pointer_cast(hashes),thrust::device_pointer_cast(hashes+n),thrust::device_pointer_cast(indices));
  reorderDataD<<<blocks,128>>>(indices,pos,vel,rho,tauA,tauB,pc,sortedActivity,originalPos,originalVel,originalRho,outA,outB,outPc,activity,n);
  findCellStartEndD<<<blocks,128,516>>>(start,end,hashes,indices,n);
  neighborSearchNum<<<blocks,128>>>(pos,rho,start,end,n,counts);thrust::exclusive_scan(thrust::device_pointer_cast(counts),thrust::device_pointer_cast(counts+n+1),thrust::device_pointer_cast(offsets));uint total;check(cudaMemcpy(&total,offsets+n,4,cudaMemcpyDeviceToHost));alloc(&neighbors,total*4);neighborSearchID<<<blocks,128>>>(pos,rho,start,end,n,offsets,neighbors);
  check(cudaMemcpy(tmpPos,pos,n*16,cudaMemcpyDeviceToDevice));check(cudaMemcpy(tmpVel,vel,n*12,cudaMemcpyDeviceToDevice));check(cudaMemcpy(tmpRho,rho,n*16,cudaMemcpyDeviceToDevice));
  for(int stage=0;stage<2;stage++){
   auto bp=stage?tmpPos:pos;auto bv=stage?tmpVel:vel;auto br=stage?tmpRho:rho;
   CfdAdamiBC_D<<<blocks,128>>>(offsets,neighbors,bp,n,acc,br,bv,flag);
   CfdCalcRHS_D<<<blocks,128>>>(deriv,bp,bv,br,offsets,neighbors,n,surface,divergence,courant,acceleration,flag);
   Calc_Shifting_D<ShiftingMethod::XSPH><<<blocks,128>>>(shifting,bp,bv,br,offsets,neighbors,n,divergence,flag);
   auto ep=stage?pos:tmpPos;auto ev=stage?vel:tmpVel;auto er=stage?rho:tmpRho;
   EulerStep_D<<<blocks,128>>>(ep,ev,er,tauA,tauB,pc,shifting,deriv,dtauA,dtauB,surface,sortedActivity,n,stage?p.dT:p.dT/2,flag);ApplyPeriodicBoundaryY_D<<<blocks,128>>>(ep,er,n);
  }
  CopySortedToOriginalWCSPH_D<<<blocks,128>>>(MarkerGroup::NON_SOLID,pos,vel,rho,tauA,tauB,pc,deriv,n,originalPos,originalVel,originalRho,outA,outB,outPc,outDeriv,indices);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaFree(neighbors));
 }
 std::ofstream output(argc>2?argv[2]:".local/chrono-loop-native.bin",std::ios::binary);dump(output,originalPos,n);dump(output,originalVel,n);dump(output,originalRho,n);dump(output,(unsigned char*)flag,4);return output?0:3;
}
