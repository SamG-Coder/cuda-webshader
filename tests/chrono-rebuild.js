// MIT host orchestration around the unchanged Chrono activity/search kernels.
// Rebuild from the current original-order GPU state, never cached neighbours.
export async function createChronoRebuild(runtime,params,n,{cudaSource,kernelFactory,gpuActiveCount=false}={}){
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
 const pool=[],bindings=[];let cursor=0,bindingCursor=0,leased=false,generation=0,diagnosticBuffers=[];
 const alloc=(input,grow=false)=>{const data=ArrayBuffer.isView(input)?input:null,size=data?data.byteLength:input,index=cursor++;let buffer=pool[index];if(buffer&&(grow?buffer.byteLength<size:buffer.byteLength!==size)){runtime.destroyBuffer(buffer);buffer=null;}if(!buffer)pool[index]=buffer=runtime.createBuffer(grow?2**Math.ceil(Math.log2(Math.max(size,4))):size);if(data)runtime.write(buffer,data);return buffer;};
 const bind=(k,data,scalars,options)=>{
  const d=k.artifact.metadata.diagnostics;if(d&&!data[d.buffer]){const buffer=alloc((2+d.capacity*d.strideWords)*4);diagnosticBuffers.push(buffer);data={...data,[d.buffer]:buffer};}
  const index=bindingCursor++,scalarBuffers=options?.scalarBuffers??{},scalarKeys=Object.keys(scalars),dataKeys=Object.keys(data);let cached=bindings[index];
  let sameShape=cached&&cached.scalarKeys.length===scalarKeys.length,changedValues=false;
  if(sameShape)for(let i=0;i<scalarKeys.length;i++){const name=scalarKeys[i];if(name!==cached.scalarKeys[i]){sameShape=false;break;}if(scalars[name]!==cached.invocation.values[name])changedValues=true;}
  const sameCounters=cached&&Object.keys(cached.scalarBuffers).length===Object.keys(scalarBuffers).length&&Object.keys(scalarBuffers).every(name=>cached.scalarBuffers[name]?.resource===scalarBuffers[name].resource&&(cached.scalarBuffers[name]?.offset??0)===(scalarBuffers[name].offset??0));
  if(!cached||cached.kernel!==k||!sameShape||!sameCounters||dataKeys.length!==cached.resourceCount||dataKeys.some(name=>cached.invocation.buffers[name]!==data[name]))bindings[index]=cached={kernel:k,scalarBuffers,scalarKeys,resourceCount:dataKeys.length,invocation:k.bind(data,scalars,options)};
  else if(changedValues)cached.invocation.setScalars(scalars);
  return cached.invocation;
 };
 const rebuild=async function(original,time,{previousStatus}={}){
  if(leased)throw Error('Release the previous neighbour state before rebuilding');leased=true;cursor=0;bindingCursor=0;diagnosticBuffers=[];const ticket=++generation;
  const clear=runtime.device.createCommandEncoder();for(const buffer of pool)clear.clearBuffer(buffer.gpuBuffer);runtime.device.queue.submit([clear.finish()]);
  const buffers={posRadD:original.pos,rhoPreMuD:original.rho,velMasD:original.vel,ad_body_D:alloc(104),ad_node1D_D:alloc(104),ad_node2D_D:alloc(104),pos_bodies_D:alloc(12),pos_nodes1D_D:alloc(12),pos_nodes2D_D:alloc(12),activityIdentifierD:alloc(n*4),extendedActivityIdD:alloc(n*4)};
  const bits=new ArrayBuffer(8);new DataView(bits).setFloat64(0,time,true);const words=new Uint32Array(bits);
  const positive=alloc(n*4),prefix=alloc(n*4),activeList=alloc(n*4),total=alloc(4);
  try{
   runtime.batch().dispatch(bind(activity,buffers,{...params,has_ad:false,'time.lo':words[0],'time.hi':words[1],'constant.countersD.numAllMarkers.lo':n,'constant.countersD.numFluidMarkers.lo':16731}),[Math.ceil(n/128)]).dispatch(bind(normalize,{activity:buffers.extendedActivityIdD,positive},{n}),[Math.ceil(n/128)]).submit();
   await runtime.exclusiveScan(positive,prefix,{count:n,total,waitForCompletion:false});
   runtime.batch().dispatch(bind(fill,{prefixSum:prefix,extendedActivityIdD:buffers.extendedActivityIdD,activeListD:activeList},{numAllMarkers:n}),[Math.ceil(n/128)]).submit();
   if(gpuActiveCount){const result=await search(buffers,activeList,n,total,previousStatus);result.dispose=()=>{if(ticket===generation)leased=false;};return result;}
   let selected;
   if(previousStatus){
    const bytes=4+previousStatus.byteLength,control=alloc(bytes),encoder=runtime.device.createCommandEncoder();
    encoder.copyBufferToBuffer(total.gpuBuffer,0,control.gpuBuffer,0,4);encoder.copyBufferToBuffer(previousStatus.gpuBuffer,0,control.gpuBuffer,4,previousStatus.byteLength);runtime.device.queue.submit([encoder.finish()]);
    const words=await runtime.read(control,Uint32Array);if(words.subarray(1).some(Boolean))throw Error('Chrono integration diagnostic: '+JSON.stringify([...words.subarray(1)]));selected=words[0];
   }else selected=(await runtime.read(total,Uint32Array))[0];
   const result=await search(buffers,activeList,selected);
   result.dispose=()=>{if(ticket===generation)leased=false;};
   return result;
  }catch(error){rebuild.dispose();throw error;}
 };
 rebuild.dispose=()=>{bindings.length=0;for(const buffer of pool)runtime.destroyBuffer(buffer);pool.length=0;leased=false;};
 return rebuild;
 async function search(buffers,activeList,n,activeCount,previousStatus){
  // With a GPU count, allocate/dispatch for the original capacity. Unchanged
  // kernels receive the true count via GPU parameters and guard inactive lanes.
  // Sentinel hashes sort the unused tail after the active records; zeroed tail
  // neighbour counts let the capacity-sized scan produce the same total.
  const p=params,c={n:originalCount};
  const cells=p['constant.paramsD.gridSize.x']*p['constant.paramsD.gridSize.y']*p['constant.paramsD.gridSize.z'];
  const b={hashes:alloc(activeCount?new Uint32Array(n).fill(0xffffffff):Math.max(n,1)*4),indices:alloc(Math.max(n,1)*4),sortedPosRad:alloc(Math.max(n,1)*16),unused:alloc(12),sortedVel:alloc(Math.max(n,1)*12),sortedRho:alloc(Math.max(n,1)*16),sortedActivity:alloc(Math.max(n,1)*4),map:alloc(new Uint32Array(c.n).fill(0xffffffff)),stress1:alloc(12),stress2:alloc(12),stress3:alloc(12),diagnostics:alloc((2+diagnostic.capacity*diagnostic.strideWords)*4),start:alloc(cells*4),end:alloc(cells*4),counts:alloc((n+1)*4),offsets:alloc((n+1)*4),total:alloc(4)};let neighbors,pendingBatch;
  const flush=()=>{if(pendingBatch){pendingBatch.submit();pendingBatch=null;}};
  const dispatch=(entry,args)=>{if(!n)return;const k=kernels[entry],values=Object.fromEntries(k.artifact.metadata.scalars.map(s=>[s.name,s.origin==='constant'?p[s.name]:n]));const scalarBuffers=activeCount?Object.fromEntries(k.artifact.metadata.scalars.filter(s=>s.origin!=='constant').map(s=>[s.name,{resource:activeCount}])):{};(pendingBatch??=runtime.batch()).dispatch(bind(k,args,values,{scalarBuffers}),[Math.ceil(n/128)]);};

  try{
   const before={...runtime.stats};
   dispatch('hashSelected',{positions:buffers.posRadD,activeList,hashes:b.hashes,indices:b.indices});flush();if(n)await runtime.sortPairs(b.hashes,b.indices,{count:n,waitForCompletion:false});
   dispatch('OriginalToSortedD',{mapOriginalToSorted:b.map,gridMarkerIndex:b.indices});
   dispatch('reorderDataD',{gridMarkerIndexD:b.indices,sortedPosRadD:b.sortedPosRad,sortedVelMasD:b.sortedVel,sortedRhoPreMuD:b.sortedRho,sortedTauXxYyZzD:b.stress1,sortedTauXyXzYzD:b.stress2,sortedPcEvSvD:b.stress3,activityIdentifierSortedD:b.sortedActivity,posRadD:buffers.posRadD,velMasD:buffers.velMasD,rhoPresMuD:buffers.rhoPreMuD,tauXxYyZzD:b.unused,tauXyXzYzD:b.unused,pcEvSvD:b.unused,activityIdentifierOriginalD:buffers.activityIdentifierD,[diagnostic.buffer]:b.diagnostics});
   dispatch('findCellStartEndD',{cellStartD:b.start,cellEndD:b.end,gridMarkerHashD:b.hashes,gridMarkerIndexD:b.indices});
   const search={sortedPosRad:b.sortedPosRad,sortedRhoPreMu:b.sortedRho,cellStart:b.start,cellEnd:b.end};
   dispatch('neighborSearchNum',{...search,numNeighborsPerPart:b.counts});flush();await runtime.exclusiveScan(b.counts,b.offsets,{count:n+1,total:b.total,waitForCompletion:false});
   let total,selected=n;
   const statusBytes=previousStatus?.byteLength??0;
   if(activeCount){
    const control=alloc(8+statusBytes),encoder=runtime.device.createCommandEncoder();encoder.copyBufferToBuffer(activeCount.gpuBuffer,0,control.gpuBuffer,0,4);encoder.copyBufferToBuffer(b.total.gpuBuffer,0,control.gpuBuffer,4,4);if(previousStatus)encoder.copyBufferToBuffer(previousStatus.gpuBuffer,0,control.gpuBuffer,8,statusBytes);runtime.device.queue.submit([encoder.finish()]);
    const words=await runtime.read(control,Uint32Array);selected=words[0];total=words[1];if(selected>n)throw Error('Active count exceeds original marker capacity');if(words.subarray(2).some(Boolean))throw Error('Chrono integration diagnostic: '+JSON.stringify([...words.subarray(2)]));
   }else total=(await runtime.read(b.total,Uint32Array))[0];
   // Counts still rebuild every step; retain capacity when only the list length
   // changes. The CUDA offsets delimit valid entries, not the allocation size.
   neighbors=alloc(Math.max(total,1)*4,true);dispatch('neighborSearchID',{...search,numNeighborsPerPart:b.offsets,neighborList:neighbors});flush();
   const readback=runtime.stats.readbackBytes-before.readbackBytes;if(readback!==(activeCount?8+statusBytes:4)||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer in selected neighbour stages');
   return {buffers:b,neighbors,count:selected,neighborEntries:total,diagnosticBuffers:[...diagnosticBuffers,b.diagnostics],dispose(){leased=false;}};

  }catch(error){if(pendingBatch&&!pendingBatch.ended)pendingBatch.discard();throw error;}
 }
}
