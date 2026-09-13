import {checkChronoCompact} from './chrono-compact-gpu.js';
export async function checkChronoSelected(runtime,{afterNeighbors}={}){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('chrono-search.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),cases=await(await load('../reports/chrono-selected-native.json')).json(),native=await(await load('../reports/chrono-selected-native.bin')).arrayBuffer();
 if(cases.length!==14||cases.at(-1).offset+cases.at(-1).sections.reduce((a,b)=>a+b,0)*4!==native.byteLength)throw Error('Incomplete selected-neighbour reference');
 const kernels={};for(const entry of ['hashSelected','findCellStartEndD','neighborSearchNum','neighborSearchID'])kernels[entry]=await runtime.kernel(source,{entry,workgroupSize:[128],...(entry==='findCellStartEndD'?{sharedMemoryBytes:516,predicatedReturns:true}:{})});
 kernels.OriginalToSortedD=await runtime.kernel(await(await load('chrono-neighbors.cu')).text(),{entry:'OriginalToSortedD',workgroupSize:[128]});
 kernels.reorderDataD=await runtime.kernel(await(await load('chrono-reorder.cu')).text(),{entry:'reorderDataD',defines:{__CUDA_ARCH__:1},workgroupSize:[128]});
 const diagnostic=kernels.reorderDataD.artifact.metadata.diagnostics;
 if(params['constant.paramsD.physics_problem']!==0)throw Error('This connected fixture requires CFD');
 let compared=0,completed=0,neighborEntries=0,intermediateReadbackBytes=0;
 const compact=await checkChronoCompact(runtime,{afterCompact:async({caseInfo:c,buffers,activeList,count:n})=>{
  const ref=cases[c.mode];if(!ref||ref.mode!==c.mode||ref.n!==n)throw Error('Selected-neighbour reference mismatch');
  const p={...params};if(c.mode){for(const axis of 'xyz'){p['constant.paramsD.worldOrigin.'+axis]=-1;p['constant.paramsD.boxDims.'+axis]=2;p['constant.paramsD.gridSize.'+axis]=10;p['constant.paramsD.cellSize.'+axis]=Math.fround(.2);}for(const [i,axis] of [...'xyz'].entries())p['constant.paramsD.'+axis+'_periodic']=!!c.periodic[i];}
  const cells=p['constant.paramsD.gridSize.x']*p['constant.paramsD.gridSize.y']*p['constant.paramsD.gridSize.z'];if(cells!==ref.cells)throw Error('Native selected grid mismatch');
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
   const total=(await runtime.read(b.total,Uint32Array))[0];if(total!==ref.neighbors)throw Error('Selected neighbour total mismatch');
   if(JSON.stringify(ref.sections)!==JSON.stringify([n,n,cells,cells,n+1,n+1,total,n*4,c.n,n*3,n*4,n]))throw Error('Selected-neighbour section sizes differ');
   neighbors=runtime.createBuffer(Math.max(total,1)*4);dispatch('neighborSearchID',{...search,numNeighborsPerPart:b.offsets,neighborList:neighbors});await runtime.idle();
   const readback=runtime.stats.readbackBytes-before.readbackBytes;if(readback!==4||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer in selected neighbour stages');intermediateReadbackBytes+=readback;
   if(afterNeighbors)await afterNeighbors({caseInfo:c,buffers:b,neighbors,count:n});
   let offset=ref.offset;for(const [i,buffer] of [b.hashes,b.indices,b.start,b.end,b.counts,b.offsets,neighbors,b.sortedPosRad,b.map,b.sortedVel,b.sortedRho,b.sortedActivity].entries()){
    const words=ref.sections[i];if(!words)continue;const actual=await runtime.read(buffer,Uint32Array),expected=new Uint32Array(native,offset,words);
    for(let j=0;j<words;j++){if(actual[j]!==expected[j])throw Error(`Selected neighbour case ${c.mode}, section ${i}, word ${j}: ${actual[j]} != ${expected[j]}`);compared++;}offset+=words*4;
   }
   const messages=await runtime.read(b.diagnostics,Uint32Array);if(messages[0]||messages[1])throw Error("Unexpected original reorder diagnostic");
   completed++;neighborEntries+=total;
  }finally{for(const buffer of [...Object.values(b),neighbors])if(buffer)runtime.destroyBuffer(buffer);}
 }});
 if(completed!==14)throw Error('Incomplete selected-neighbour coverage');
 return {cases:completed,compared,neighborEntries,nativeNormalizedExact:true,originalMarkerIdsPreserved:true,activityAndCompactionVerified:compact.nativeNormalizedExact,intermediateReadbackBytes:intermediateReadbackBytes+compact.intermediateReadbackBytes,cpuPhysics:false,neighborsConnected:true,originalPropertyReorderConnected:true,nativeInitialProperties:true,fullSolverAdvanced:false};
}
