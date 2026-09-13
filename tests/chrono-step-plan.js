// MIT orchestration of the original Chrono RK2 kernels.
export async function createChronoStep(runtime,params,{cudaSource,kernelFactory,deferDiagnostics=false}={}){
 if(params['constant.paramsD.physics_problem']!==0||params['constant.paramsD.shifting_method']!==2)throw Error('Chrono step requires CFD with XSPH');
 const load=async p=>await(await fetch(new URL(p,import.meta.url))).text();
 const integration=cudaSource??await load('chrono-rk2.cu'),shiftSource=cudaSource??await load('chrono-shifting.cu'),rhsSource=cudaSource??await load('chrono-rhs.cu'),bcSource=cudaSource??await load('chrono-adami.cu');
 const make=(source,entry)=>(kernelFactory??((s,o)=>runtime.kernel(s,o)))(source,{entry,defines:{__CUDA_ARCH__:1},workgroupSize:[128]});
 const scatter=await make(cudaSource??(integration+'\n'+await load('chrono-scatter.cuh')),'CopySortedToOriginalWCSPH_D');
 const euler=await make(integration,'EulerStep_D'),periodic=await make(integration,'ApplyPeriodicBoundaryY_D'),shift=await make(shiftSource,'Calc_Shifting_D<ShiftingMethod::XSPH>'),bc=await make(bcSource,'CfdAdamiBC_D'),rhs=await make(rhsSource,'CfdCalcRHS_D');
 const pool=[];
 const step=async function(original,{buffers:b,neighbors,count:n,diagnosticBuffers:preparationDiagnostics=[]}){
  if(!n)return;
  const originalCount=original.pos.byteLength/16;
  let cursor=0;
  const clear=runtime.device.createCommandEncoder();for(const buffer of pool)clear.clearBuffer(buffer.gpuBuffer);runtime.device.queue.submit([clear.finish()]);
  const alloc=size=>{const index=cursor++;let buffer=pool[index];if(buffer&&buffer.byteLength!==size){runtime.destroyBuffer(buffer);buffer=null;}if(!buffer)pool[index]=buffer=runtime.createBuffer(size);return buffer;},state=()=>({pos:alloc(n*16),vel:alloc(n*12),rho:alloc(n*16)}),y=state(),tmp=state(),flag=alloc(4),shifting=alloc(n*12),acc=alloc(n*12),tauA=alloc(12),tauB=alloc(12),pc=alloc(12),dtauA=alloc(12),dtauB=alloc(12),deriv=alloc(n*16),surface=alloc(n*4),divergence=alloc(n*4),courant=alloc(n*4),acceleration=alloc(n*4),diags=new Map();
  for(const k of [euler,periodic,shift,bc,rhs,scatter])if(k.artifact.metadata.diagnostics){const d=k.artifact.metadata.diagnostics;diags.set(k,alloc((2+d.capacity*d.strideWords)*4));}
  const bind=(k,data,extra={})=>{const d=k.artifact.metadata.diagnostics;if(d)data={...data,[d.buffer]:diags.get(k)};return k.bind(Object.fromEntries(k.artifact.metadata.bindings.map(x=>{if(!data[x.name])throw Error('Missing '+x.name);return [x.name,data[x.name]]})),{...params,numActive:n,...extra});};
  const shiftBind=(s,div)=>bind(shift,{vel_XSPH_Sorted_D:shifting,sortedPosRad:s.pos,sortedVelMas:s.vel,sortedRhoPreMu:s.rho,numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedPosDivergence:div,error_flag:flag});
  const eulerBind=(s,d,surf,dt)=>bind(euler,{posRadD:s.pos,velMasD:s.vel,rhoPresMuD:s.rho,tauXxYyZzD:tauA,tauXyXzYzD:tauB,pcEvSvD:pc,vel_XSPH_D:shifting,derivVelRhoD:d,derivTauXxYyZzD:dtauA,derivTauXyXzYzD:dtauB,freeSurfaceIdD:surf,activityIdentifierSortedD:b.sortedActivity,error_flag:flag},{dT:dt});
  const periodicBind=s=>bind(periodic,{posRadD:s.pos,rhoPresMuD:s.rho});
  const copy=(src,dests)=>{const enc=runtime.device.createCommandEncoder();for(const dst of dests)for(const [name,bytes]of [['pos',n*16],['vel',n*12],['rho',n*16]])enc.copyBufferToBuffer(src[name].gpuBuffer,0,dst[name].gpuBuffer,0,bytes);runtime.device.queue.submit([enc.finish()]);};
  try{
   const before={...runtime.stats},grid=[Math.ceil(n/128)],dt=params['constant.paramsD.dT'];copy({pos:b.sortedPosRad,vel:b.sortedVel,rho:b.sortedRho},[y,tmp]);
   runtime.batch().dispatch(bind(bc,{numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedPosRadD:y.pos,bceAcc:acc,sortedRhoPresMuD:y.rho,sortedVelMasD:y.vel,error_flag:flag}),grid).dispatch(bind(rhs,{sortedDerivVelRho:deriv,sortedPosRad:y.pos,sortedVelMas:y.vel,sortedRhoPreMu:y.rho,numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedFreeSurfaceIdD:surface,sortedPosDivergence:divergence,courantViscousTimeStep:courant,accelerationTimeStep:acceleration,error_flag:flag}),grid).submit();
   runtime.batch().dispatch(shiftBind(y,divergence),grid).dispatch(eulerBind(tmp,deriv,surface,dt/2),grid).dispatch(periodicBind(tmp),grid).submit();
   runtime.batch().dispatch(bind(bc,{numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedPosRadD:tmp.pos,bceAcc:acc,sortedRhoPresMuD:tmp.rho,sortedVelMasD:tmp.vel,error_flag:flag}),grid)
    .dispatch(bind(rhs,{sortedDerivVelRho:deriv,sortedPosRad:tmp.pos,sortedVelMas:tmp.vel,sortedRhoPreMu:tmp.rho,numNeighborsPerPart:b.offsets,neighborList:neighbors,sortedFreeSurfaceIdD:surface,sortedPosDivergence:divergence,courantViscousTimeStep:courant,accelerationTimeStep:acceleration,error_flag:flag}),grid)
    .dispatch(shiftBind(tmp,divergence),grid).dispatch(eulerBind(y,deriv,surface,dt),grid).dispatch(periodicBind(y),grid).submit();
   if(runtime.stats.readbackBytes!==before.readbackBytes||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('CPU transfer during RK2');
   const stresses=Array.from({length:6},(_,i)=>alloc((i<3?n:originalCount)*12)),originalDeriv=alloc(originalCount*16);
   const transferBefore={...runtime.stats};
   runtime.batch().dispatch(bind(scatter,{sortedPosRad:y.pos,sortedVelMas:y.vel,sortedRhoPresMu:y.rho,sortedTauXxYyZz:stresses[0],sortedTauXyXXzYz:stresses[1],sortedPcEvSv:stresses[2],derivVelRho:deriv,posRadOriginal:original.pos,velMasOriginal:original.vel,rhoPresMuOriginal:original.rho,tauXxYyZzOriginal:stresses[3],tauXyXzYzOriginal:stresses[4],pcEvSvOriginal:stresses[5],derivVelRhoOriginal:originalDeriv,gridMarkerIndex:b.indices},{group:4}),grid).submit();
   if(runtime.stats.readbackBytes!==transferBefore.readbackBytes||runtime.stats.dataBytesUploaded!==transferBefore.dataBytesUploaded)throw Error('CPU transfer during original copy-back');

   // Gather every status word into one readback. Queue ordering guarantees the
   // copy follows all kernels, so no separate idle wait or per-kernel map is needed.
   const allDiagnostics=[...diags.values(),...preparationDiagnostics],status=alloc(4+allDiagnostics.length*8),enc=runtime.device.createCommandEncoder();
   enc.copyBufferToBuffer(flag.gpuBuffer,0,status.gpuBuffer,0,4);
   let offset=4;for(const d of allDiagnostics){enc.copyBufferToBuffer(d.gpuBuffer,0,status.gpuBuffer,offset,8);offset+=8;}
   runtime.device.queue.submit([enc.finish()]);
   // The loop must consume this snapshot before the next step clears its pool.
   if(deferDiagnostics)return status;
   const errors=await runtime.read(status,Uint32Array);if(errors.some(Boolean))throw Error('Chrono integration diagnostic: '+JSON.stringify([...errors]));
  }catch(error){step.dispose();throw error;}
 };
 step.dispose=()=>{for(const buffer of pool)runtime.destroyBuffer(buffer);pool.length=0;};
 return step;
}
