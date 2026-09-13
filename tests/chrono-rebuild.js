// MIT host orchestration around the unchanged Chrono activity/search kernels.
// Rebuild from the current original-order GPU state, never cached neighbours.
export async function createChronoRebuild(runtime,params,n){
 const load=path=>fetch(new URL(path,import.meta.url));
 const originalCount=n;
 const source=await(await load('chrono-search.cu')).text();
 const kernels={};for(const entry of ['hashSelected','findCellStartEndD','neighborSearchNum','neighborSearchID'])kernels[entry]=await runtime.kernel(source,{entry,workgroupSize:[128],...(entry==='findCellStartEndD'?{sharedMemoryBytes:516,predicatedReturns:true}:{})});
 kernels.OriginalToSortedD=await runtime.kernel(await(await load('chrono-neighbors.cu')).text(),{entry:'OriginalToSortedD',workgroupSize:[128]});
 kernels.reorderDataD=await runtime.kernel(await(await load('chrono-reorder.cu')).text(),{entry:'reorderDataD',defines:{__CUDA_ARCH__:1},workgroupSize:[128]});
 const diagnostic=kernels.reorderDataD.artifact.metadata.diagnostics;
 if(params['constant.paramsD.physics_problem']!==0)throw Error('This connected fixture requires CFD');

 const activity=await runtime.kernel(await(await load('chrono-activity.cu')).text(),{entry:'UpdateActivityD',workgroupSize:[128]});
 const compactSource=await(await load('chrono-compact.cu')).text();
 const normalize=await runtime.kernel(compactSource,{entry:'normalizeActivity',workgroupSize:[128]}),fill=await runtime.kernel(compactSource,{entry:'fillActiveListD',workgroupSize:[128]});
 return async function rebuild(original,time){
  const owned=[],alloc=bytes=>{const b=runtime.createBuffer(bytes);owned.push(b);return b;};
  const buffers={posRadD:original.pos,rhoPreMuD:original.rho,velMasD:original.vel,ad_body_D:alloc(104),ad_node1D_D:alloc(104),ad_node2D_D:alloc(104),pos_bodies_D:alloc(12),pos_nodes1D_D:alloc(12),pos_nodes2D_D:alloc(12),activityIdentifierD:alloc(n*4),extendedActivityIdD:alloc(n*4)};
  const bits=new ArrayBuffer(8);new DataView(bits).setFloat64(0,time,true);const words=new Uint32Array(bits);
  const positive=alloc(n*4),prefix=alloc(n*4),activeList=alloc(n*4),total=alloc(4);
  try{
   runtime.batch().dispatch(activity.bind(buffers,{...params,has_ad:false,'time.lo':words[0],'time.hi':words[1],'constant.countersD.numAllMarkers.lo':n,'constant.countersD.numFluidMarkers.lo':16731}),[Math.ceil(n/128)]).dispatch(normalize.bind({activity:buffers.extendedActivityIdD,positive},{n}),[Math.ceil(n/128)]).submit();
   await runtime.exclusiveScan(positive,prefix,{count:n,total});
   runtime.batch().dispatch(fill.bind({prefixSum:prefix,extendedActivityIdD:buffers.extendedActivityIdD,activeListD:activeList},{numAllMarkers:n}),[Math.ceil(n/128)]).submit();
   const selected=(await runtime.read(total,Uint32Array))[0];
   const result=await search(buffers,activeList,selected);
   const dispose=result.dispose;result.dispose=()=>{dispose();for(const b of owned)runtime.destroyBuffer(b);};
   return result;
  }catch(error){for(const b of owned)runtime.destroyBuffer(b);throw error;}
 };
 async function search(buffers,activeList,n){
  const p=params,c={n:originalCount};
  const cells=p['constant.paramsD.gridSize.x']*p['constant.paramsD.gridSize.y']*p['constant.paramsD.gridSize.z'];
  const b={hashes:runtime.createBuffer(Math.max(n,1)*4),indices:runtime.createBuffer(Math.max(n,1)*4),sortedPosRad:runtime.createBuffer(Math.max(n,1)*16),unused:runtime.createBuffer(12),sortedVel:runtime.createBuffer(Math.max(n,1)*12),sortedRho:runtime.createBuffer(Math.max(n,1)*16),sortedActivity:runtime.createBuffer(Math.max(n,1)*4),map:runtime.createBuffer(new Uint32Array(c.n).fill(0xffffffff)),stress1:runtime.createBuffer(12),stress2:runtime.createBuffer(12),stress3:runtime.createBuffer(12),diagnostics:runtime.createBuffer((2+diagnostic.capacity*diagnostic.strideWords)*4),start:runtime.createBuffer(cells*4),end:runtime.createBuffer(cells*4),counts:runtime.createBuffer((n+1)*4),offsets:runtime.createBuffer((n+1)*4),total:runtime.createBuffer(4)};let neighbors;
  const dispatch=(entry,args)=>{if(!n)return;const k=kernels[entry],values=Object.fromEntries(k.artifact.metadata.scalars.map(s=>[s.name,s.origin==='constant'?p[s.name]:n]));runtime.batch().dispatch(k.bind(args,values),[Math.ceil(n/128)]).submit();};

  try{
   const before={...runtime.stats};
   dispatch('hashSelected',{positions:buffers.posRadD,activeList,hashes:b.hashes,indices:b.indices});if(n)await runtime.sortPairs(b.hashes,b.indices,{count:n});
   dispatch('OriginalToSortedD',{mapOriginalToSorted:b.map,gridMarkerIndex:b.indices});
   dispatch('reorderDataD',{gridMarkerIndexD:b.indices,sortedPosRadD:b.sortedPosRad,sortedVelMasD:b.sortedVel,sortedRhoPreMuD:b.sortedRho,sortedTauXxYyZzD:b.stress1,sortedTauXyXzYzD:b.stress2,sortedPcEvSvD:b.stress3,activityIdentifierSortedD:b.sortedActivity,posRadD:buffers.posRadD,velMasD:buffers.velMasD,rhoPresMuD:buffers.rhoPreMuD,tauXxYyZzD:b.unused,tauXyXzYzD:b.unused,pcEvSvD:b.unused,activityIdentifierOriginalD:buffers.activityIdentifierD,[diagnostic.buffer]:b.diagnostics});
   dispatch('findCellStartEndD',{cellStartD:b.start,cellEndD:b.end,gridMarkerHashD:b.hashes,gridMarkerIndexD:b.indices});
   const search={sortedPosRad:b.sortedPosRad,sortedRhoPreMu:b.sortedRho,cellStart:b.start,cellEnd:b.end};
   dispatch('neighborSearchNum',{...search,numNeighborsPerPart:b.counts});await runtime.exclusiveScan(b.counts,b.offsets,{count:n+1,total:b.total});
   const total=(await runtime.read(b.total,Uint32Array))[0];
   neighbors=runtime.createBuffer(Math.max(total,1)*4);dispatch('neighborSearchID',{...search,numNeighborsPerPart:b.offsets,neighborList:neighbors});await runtime.idle();
   const readback=runtime.stats.readbackBytes-before.readbackBytes;if(readback!==4||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer in selected neighbour stages');
   return {buffers:b,neighbors,count:n,neighborEntries:total,dispose(){for(const v of [...Object.values(b),neighbors])runtime.destroyBuffer(v);}};

  }catch(error){for(const v of [...Object.values(b),neighbors])if(v)runtime.destroyBuffer(v);throw error;}
 }
}
