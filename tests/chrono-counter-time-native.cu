// MIT native validation harness; the included Counters declaration retains Chrono's BSD license.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "chrono-counter-time.cu"
static_assert(sizeof(size_t)==8 && sizeof(Counters)==176, "Expected native 64-bit Counters ABI");
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
int main(){
 unsigned int* out;check(cudaMalloc(&out,96*4));
 const double times[]={0.0,0.5-0x1p-40,0.5,0.5+0x1p-40,-0.5,-0.0,0x1p-100,-0x1p-100,std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),1.0};
 const size_t values[]={0,1,30327,0xffffffffULL,0x100000000ULL,0x100000001ULL,0x1fffffffffffffULL,0x20000000000001ULL,0x8000000000000000ULL,0xffffffff00000000ULL,0xffffffffffffffffULL,16731};
 printf("[\n");
 for(int c=0;c<12;c++){
  Counters counters{};
  counters.numFsiBodies=values[(0+c)%12];
  counters.numFsiMeshes1D=values[(1+c)%12];
  counters.numFsiMeshes2D=values[(2+c)%12];
  counters.numFsiNodes1D=values[(3+c)%12];
  counters.numFsiNodes2D=values[(4+c)%12];
  counters.numFsiElements1D=values[(5+c)%12];
  counters.numFsiElements2D=values[(6+c)%12];
  counters.numGhostMarkers=values[(7+c)%12];
  counters.numHelperMarkers=values[(8+c)%12];
  counters.numFluidMarkers=values[(9+c)%12];
  counters.numBoundaryMarkers=values[(10+c)%12];
  counters.numRigidMarkers=values[(11+c)%12];
  counters.numMesh1DMarkers=values[(12+c)%12];
  counters.numMesh2DMarkers=values[(13+c)%12];
  counters.numBceMarkers=values[(14+c)%12];
  counters.numAllMarkers=values[(15+c)%12];
  counters.startBoundaryMarkers=values[(16+c)%12];
  counters.startRigidMarkers=values[(17+c)%12];
  counters.startMesh1DMarkers=values[(18+c)%12];
  counters.startMesh2DMarkers=values[(19+c)%12];
  counters.numActiveParticles=values[(20+c)%12];
  counters.numExtendedParticles=values[(21+c)%12];
  check(cudaMemcpyToSymbol(countersD,&counters,sizeof(counters)));
  counterTimeProbe<<<1,1>>>(out,times[c]);check(cudaGetLastError());check(cudaDeviceSynchronize());
  unsigned int words[96],time[2];check(cudaMemcpy(words,out,sizeof(words),cudaMemcpyDeviceToHost));memcpy(time,&times[c],8);
  printf("%s{\"scalars\":{\"time.lo\":%u,\"time.hi\":%u",c?",\n":"",time[0],time[1]);
  printf(",\"constant.countersD.numFsiBodies.lo\":%u,\"constant.countersD.numFsiBodies.hi\":%u",(unsigned int)counters.numFsiBodies,(unsigned int)(counters.numFsiBodies>>32));
  printf(",\"constant.countersD.numFsiMeshes1D.lo\":%u,\"constant.countersD.numFsiMeshes1D.hi\":%u",(unsigned int)counters.numFsiMeshes1D,(unsigned int)(counters.numFsiMeshes1D>>32));
  printf(",\"constant.countersD.numFsiMeshes2D.lo\":%u,\"constant.countersD.numFsiMeshes2D.hi\":%u",(unsigned int)counters.numFsiMeshes2D,(unsigned int)(counters.numFsiMeshes2D>>32));
  printf(",\"constant.countersD.numFsiNodes1D.lo\":%u,\"constant.countersD.numFsiNodes1D.hi\":%u",(unsigned int)counters.numFsiNodes1D,(unsigned int)(counters.numFsiNodes1D>>32));
  printf(",\"constant.countersD.numFsiNodes2D.lo\":%u,\"constant.countersD.numFsiNodes2D.hi\":%u",(unsigned int)counters.numFsiNodes2D,(unsigned int)(counters.numFsiNodes2D>>32));
  printf(",\"constant.countersD.numFsiElements1D.lo\":%u,\"constant.countersD.numFsiElements1D.hi\":%u",(unsigned int)counters.numFsiElements1D,(unsigned int)(counters.numFsiElements1D>>32));
  printf(",\"constant.countersD.numFsiElements2D.lo\":%u,\"constant.countersD.numFsiElements2D.hi\":%u",(unsigned int)counters.numFsiElements2D,(unsigned int)(counters.numFsiElements2D>>32));
  printf(",\"constant.countersD.numGhostMarkers.lo\":%u,\"constant.countersD.numGhostMarkers.hi\":%u",(unsigned int)counters.numGhostMarkers,(unsigned int)(counters.numGhostMarkers>>32));
  printf(",\"constant.countersD.numHelperMarkers.lo\":%u,\"constant.countersD.numHelperMarkers.hi\":%u",(unsigned int)counters.numHelperMarkers,(unsigned int)(counters.numHelperMarkers>>32));
  printf(",\"constant.countersD.numFluidMarkers.lo\":%u,\"constant.countersD.numFluidMarkers.hi\":%u",(unsigned int)counters.numFluidMarkers,(unsigned int)(counters.numFluidMarkers>>32));
  printf(",\"constant.countersD.numBoundaryMarkers.lo\":%u,\"constant.countersD.numBoundaryMarkers.hi\":%u",(unsigned int)counters.numBoundaryMarkers,(unsigned int)(counters.numBoundaryMarkers>>32));
  printf(",\"constant.countersD.numRigidMarkers.lo\":%u,\"constant.countersD.numRigidMarkers.hi\":%u",(unsigned int)counters.numRigidMarkers,(unsigned int)(counters.numRigidMarkers>>32));
  printf(",\"constant.countersD.numMesh1DMarkers.lo\":%u,\"constant.countersD.numMesh1DMarkers.hi\":%u",(unsigned int)counters.numMesh1DMarkers,(unsigned int)(counters.numMesh1DMarkers>>32));
  printf(",\"constant.countersD.numMesh2DMarkers.lo\":%u,\"constant.countersD.numMesh2DMarkers.hi\":%u",(unsigned int)counters.numMesh2DMarkers,(unsigned int)(counters.numMesh2DMarkers>>32));
  printf(",\"constant.countersD.numBceMarkers.lo\":%u,\"constant.countersD.numBceMarkers.hi\":%u",(unsigned int)counters.numBceMarkers,(unsigned int)(counters.numBceMarkers>>32));
  printf(",\"constant.countersD.numAllMarkers.lo\":%u,\"constant.countersD.numAllMarkers.hi\":%u",(unsigned int)counters.numAllMarkers,(unsigned int)(counters.numAllMarkers>>32));
  printf(",\"constant.countersD.startBoundaryMarkers.lo\":%u,\"constant.countersD.startBoundaryMarkers.hi\":%u",(unsigned int)counters.startBoundaryMarkers,(unsigned int)(counters.startBoundaryMarkers>>32));
  printf(",\"constant.countersD.startRigidMarkers.lo\":%u,\"constant.countersD.startRigidMarkers.hi\":%u",(unsigned int)counters.startRigidMarkers,(unsigned int)(counters.startRigidMarkers>>32));
  printf(",\"constant.countersD.startMesh1DMarkers.lo\":%u,\"constant.countersD.startMesh1DMarkers.hi\":%u",(unsigned int)counters.startMesh1DMarkers,(unsigned int)(counters.startMesh1DMarkers>>32));
  printf(",\"constant.countersD.startMesh2DMarkers.lo\":%u,\"constant.countersD.startMesh2DMarkers.hi\":%u",(unsigned int)counters.startMesh2DMarkers,(unsigned int)(counters.startMesh2DMarkers>>32));
  printf(",\"constant.countersD.numActiveParticles.lo\":%u,\"constant.countersD.numActiveParticles.hi\":%u",(unsigned int)counters.numActiveParticles,(unsigned int)(counters.numActiveParticles>>32));
  printf(",\"constant.countersD.numExtendedParticles.lo\":%u,\"constant.countersD.numExtendedParticles.hi\":%u",(unsigned int)counters.numExtendedParticles,(unsigned int)(counters.numExtendedParticles>>32));
  printf("},\"output\":[");for(int i=0;i<96;i++)printf("%s%u",i?",":"",words[i]);printf("]}");
 }
 printf("\n]\n");check(cudaFree(out));
}
