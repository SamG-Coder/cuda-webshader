// MIT host orchestration around the unchanged Chrono activity/search kernels.
// Rebuild from the current original-order GPU state, never cached neighbours.
export async function createChronoRebuild(runtime,params,n,{cudaSource,kernelFactory}={}){
 const make=kernelFactory??((source,options)=>runtime.kernel(source,options));
 const load=path=>fetch(new URL(path,import.meta.url));
 const originalCount=n;
 const source=cudaSource??await(await load('chrono-search.cu')).text();
 const kernels={};for(const entry of ['hashSelected','findCellStartEndD','neighborSearchNum','neighborSearchID'])kernels[entry]=await make(source,{entry,workgroupSize:[128],...(entry==='findCellStartEndD'?{sharedMemoryBytes:516,predicatedReturns:true}:{})});
 kernels.OriginalToSortedD=await make(cudaSource??await(await load('chrono-neighbors.cu')).text(),{entry:'OriginalToSortedD',workgroupSize:[128]});
 kernels.reorderDataD=await make(cudaSource??await(await load('chrono-reorder.cu')).text(),{entry:'reorderDataD',defines:{__CUDA_ARCH__:1},workgroupSize:[128]});
 const diagnostic=kernels.reorderDataD.artifact.metadata.diagnostics;
 if(params['constant.paramsD.physics_problem']!==0)throw Error('This connected fixture requires CFD');

 const activity=await make(cudaSource??await(await load('chrono-activity.cu')).text(),{entry:'UpdateActivityD',workgroupSize:[128]});
 const compactSource=cudaSource??await(await load('chrono-compact.cu')).text();
 const normalize=await make(compactSource,{entry:'normalizeActivity',workgroupSize:[128]}),fill=await make(compactSource,{entry:'fillActiveListD',workgroupSize:[128]});
 const pool=[];let cursor=0,leased=false,generation=0,diagnosticBuffers=[];
 const alloc=input=>{const data=ArrayBuffer.isView(input)?input:null,size=data?data.byteLength:input,index=cursor++;let buffer=pool[index];if(buffer&&buffer.byteLength!==size){runtime.destroyBuffer(buffer);buffer=null;}if(!buffer)pool[index]=buffer=runtime.createBuffer(size);if(data)runtime.write(buffer,data);return buffer;};
 const bind=(k,data,scalars)=>{const d=k.artifact.metadata.diagnostics;if(d&&!data[d.buffer]){const buffer=alloc((2+d.capacity*d.strideWords)*4);diagnosticBuffers.push(buffer);data={...data,[d.buffer]:buffer};}return k.bind(data,scalars);};
 const rebuild=async function(original,time){
  if(leased)throw Error('Release the previous neighbour state before rebuilding');leased=true;cursor=0;diagnosticBuffers=[];const ticket=++generation;
  const clear=runtime.device.createCommandEncoder();for(const buffer of pool)clear.clearBuffer(buffer.gpuBuffer);runtime.device.queue.submit([clear.finish()]);
  const buffers={posRadD:original.pos,rhoPreMuD:original.rho,velMasD:original.vel,ad_body_D:alloc(104),ad_node1D_D:alloc(104),ad_node2D_D:alloc(104),pos_bodies_D:alloc(12),pos_nodes1D_D:alloc(12),pos_nodes2D_D:alloc(12),activityIdentifierD:alloc(n*4),extendedActivityIdD:alloc(n*4)};
  const bits=new ArrayBuffer(8);new DataView(bits).setFloat64(0,time,true);const words=new Uint32Array(bits);
  const positive=alloc(n*4),prefix=alloc(n*4),activeList=alloc(n*4),total=alloc(4);
  try{
   runtime.batch().dispatch(bind(activity,buffers,{...params,has_ad:false,'time.lo':words[0],'time.hi':words[1],'constant.countersD.numAllMarkers.lo':n,'constant.countersD.numFluidMarkers.lo':16731}),[Math.ceil(n/128)]).dispatch(bind(normalize,{activity:buffers.extendedActivityIdD,positive},{n}),[Math.ceil(n/128)]).submit();
   await runtime.exclusiveScan(positive,prefix,{count:n,total,waitForCompletion:false});
   runtime.batch().dispatch(bind(fill,{prefixSum:prefix,extendedActivityIdD:buffers.extendedActivityIdD,activeListD:activeList},{numAllMarkers:n}),[Math.ceil(n/128)]).submit();
   const selected=(await runtime.read(total,Uint32Array))[0];
   const result=await search(buffers,activeList,selected);
   result.dispose=()=>{if(ticket===generation)leased=false;};
   return result;
  }catch(error){rebuild.dispose();throw error;}
 };
 rebuild.dispose=()=>{for(const buffer of pool)runtime.destroyBuffer(buffer);pool.length=0;leased=false;};
 return rebuild;
 async function search(buffers,activeList,n){
  const p=params,c={n:originalCount};
  const cells=p['constant.paramsD.gridSize.x']*p['constant.paramsD.gridSize.y']*p['constant.paramsD.gridSize.z'];
  const b={hashes:alloc(Math.max(n,1)*4),indices:alloc(Math.max(n,1)*4),sortedPosRad:alloc(Math.max(n,1)*16),unused:alloc(12),sortedVel:alloc(Math.max(n,1)*12),sortedRho:alloc(Math.max(n,1)*16),sortedActivity:alloc(Math.max(n,1)*4),map:alloc(new Uint32Array(c.n).fill(0xffffffff)),stress1:alloc(12),stress2:alloc(12),stress3:alloc(12),diagnostics:alloc((2+diagnostic.capacity*diagnostic.strideWords)*4),start:alloc(cells*4),end:alloc(cells*4),counts:alloc((n+1)*4),offsets:alloc((n+1)*4),total:alloc(4)};let neighbors;
  const dispatch=(entry,args)=>{if(!n)return;const k=kernels[entry],values=Object.fromEntries(k.artifact.metadata.scalars.map(s=>[s.name,s.origin==='constant'?p[s.name]:n]));runtime.batch().dispatch(bind(k,args,values),[Math.ceil(n/128)]).submit();};

  try{
   const before={...runtime.stats};
   dispatch('hashSelected',{positions:buffers.posRadD,activeList,hashes:b.hashes,indices:b.indices});if(n)await runtime.sortPairs(b.hashes,b.indices,{count:n,waitForCompletion:false});
   dispatch('OriginalToSortedD',{mapOriginalToSorted:b.map,gridMarkerIndex:b.indices});
   dispatch('reorderDataD',{gridMarkerIndexD:b.indices,sortedPosRadD:b.sortedPosRad,sortedVelMasD:b.sortedVel,sortedRhoPreMuD:b.sortedRho,sortedTauXxYyZzD:b.stress1,sortedTauXyXzYzD:b.stress2,sortedPcEvSvD:b.stress3,activityIdentifierSortedD:b.sortedActivity,posRadD:buffers.posRadD,velMasD:buffers.velMasD,rhoPresMuD:buffers.rhoPreMuD,tauXxYyZzD:b.unused,tauXyXzYzD:b.unused,pcEvSvD:b.unused,activityIdentifierOriginalD:buffers.activityIdentifierD,[diagnostic.buffer]:b.diagnostics});
   dispatch('findCellStartEndD',{cellStartD:b.start,cellEndD:b.end,gridMarkerHashD:b.hashes,gridMarkerIndexD:b.indices});
   const search={sortedPosRad:b.sortedPosRad,sortedRhoPreMu:b.sortedRho,cellStart:b.start,cellEnd:b.end};
   dispatch('neighborSearchNum',{...search,numNeighborsPerPart:b.counts});await runtime.exclusiveScan(b.counts,b.offsets,{count:n+1,total:b.total,waitForCompletion:false});
   const total=(await runtime.read(b.total,Uint32Array))[0];
   neighbors=alloc(Math.max(total,1)*4);dispatch('neighborSearchID',{...search,numNeighborsPerPart:b.offsets,neighborList:neighbors});
   const readback=runtime.stats.readbackBytes-before.readbackBytes;if(readback!==4||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer in selected neighbour stages');
   return {buffers:b,neighbors,count:n,neighborEntries:total,diagnosticBuffers:[...diagnosticBuffers,b.diagnostics],dispose(){leased=false;}};

  }catch(error){throw error;}
 }
}
