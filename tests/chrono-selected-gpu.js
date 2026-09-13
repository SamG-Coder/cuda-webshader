import {checkChronoCompact} from './chrono-compact-gpu.js';
export async function checkChronoSelected(runtime){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('chrono-search.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),cases=await(await load('../reports/chrono-selected-native.json')).json(),native=await(await load('../reports/chrono-selected-native.bin')).arrayBuffer();
 if(cases.length!==14||cases.at(-1).offset+cases.at(-1).sections.reduce((a,b)=>a+b,0)*4!==native.byteLength)throw Error('Incomplete selected-neighbour reference');
 const kernels={};for(const entry of ['hashSelected','gatherMarkers','findCellStartEndD','neighborSearchNum','neighborSearchID'])kernels[entry]=await runtime.kernel(source,{entry,workgroupSize:[128],...(entry==='findCellStartEndD'?{sharedMemoryBytes:516,predicatedReturns:true}:{})});
 let compared=0,completed=0,neighborEntries=0,intermediateReadbackBytes=0;
 const compact=await checkChronoCompact(runtime,{afterCompact:async({caseInfo:c,buffers,activeList,count:n})=>{
  const ref=cases[c.mode];if(!ref||ref.mode!==c.mode||ref.n!==n)throw Error('Selected-neighbour reference mismatch');
  const p={...params};if(c.mode){for(const axis of 'xyz'){p['constant.paramsD.worldOrigin.'+axis]=-1;p['constant.paramsD.boxDims.'+axis]=2;p['constant.paramsD.gridSize.'+axis]=10;p['constant.paramsD.cellSize.'+axis]=Math.fround(.2);}for(const [i,axis] of [...'xyz'].entries())p['constant.paramsD.'+axis+'_periodic']=!!c.periodic[i];}
  const cells=p['constant.paramsD.gridSize.x']*p['constant.paramsD.gridSize.y']*p['constant.paramsD.gridSize.z'];if(cells!==ref.cells)throw Error('Native selected grid mismatch');
  const b={hashes:runtime.createBuffer(Math.max(n,1)*4),indices:runtime.createBuffer(Math.max(n,1)*4),sortedPosRad:runtime.createBuffer(Math.max(n,1)*16),unused:runtime.createBuffer(16),start:runtime.createBuffer(cells*4),end:runtime.createBuffer(cells*4),counts:runtime.createBuffer((n+1)*4),offsets:runtime.createBuffer((n+1)*4),total:runtime.createBuffer(4)};let neighbors;
  const dispatch=(entry,args)=>{if(!n)return;const k=kernels[entry],values=Object.fromEntries(k.artifact.metadata.scalars.map(s=>[s.name,s.origin==='constant'?p[s.name]:n]));runtime.batch().dispatch(k.bind(args,values),[Math.ceil(n/128)]).submit();};
  try{
   const before={...runtime.stats};
   dispatch('hashSelected',{positions:buffers.posRadD,activeList,hashes:b.hashes,indices:b.indices});if(n)await runtime.sortPairs(b.hashes,b.indices,{count:n});
   dispatch('gatherMarkers',{positions:buffers.posRadD,indices:b.indices,sortedPosRad:b.sortedPosRad});
   dispatch('findCellStartEndD',{cellStartD:b.start,cellEndD:b.end,gridMarkerHashD:b.hashes,gridMarkerIndexD:b.indices});
   const search={sortedPosRad:b.sortedPosRad,sortedRhoPreMu:b.unused,cellStart:b.start,cellEnd:b.end};
   dispatch('neighborSearchNum',{...search,numNeighborsPerPart:b.counts});await runtime.exclusiveScan(b.counts,b.offsets,{count:n+1,total:b.total});
   const total=(await runtime.read(b.total,Uint32Array))[0];if(total!==ref.neighbors)throw Error('Selected neighbour total mismatch');
   if(JSON.stringify(ref.sections)!==JSON.stringify([n,n,cells,cells,n+1,n+1,total,n*4]))throw Error('Selected-neighbour section sizes differ');
   neighbors=runtime.createBuffer(Math.max(total,1)*4);dispatch('neighborSearchID',{...search,numNeighborsPerPart:b.offsets,neighborList:neighbors});await runtime.idle();
   const readback=runtime.stats.readbackBytes-before.readbackBytes;if(readback!==4||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer in selected neighbour stages');intermediateReadbackBytes+=readback;
   let offset=ref.offset;for(const [i,buffer] of [b.hashes,b.indices,b.start,b.end,b.counts,b.offsets,neighbors,b.sortedPosRad].entries()){
    const words=ref.sections[i];if(!words)continue;const actual=await runtime.read(buffer,Uint32Array),expected=new Uint32Array(native,offset,words);
    for(let j=0;j<words;j++){if(actual[j]!==expected[j])throw Error(`Selected neighbour case ${c.mode}, section ${i}, word ${j}: ${actual[j]} != ${expected[j]}`);compared++;}offset+=words*4;
   }
   completed++;neighborEntries+=total;
  }finally{for(const buffer of [...Object.values(b),neighbors])if(buffer)runtime.destroyBuffer(buffer);}
 }});
 if(completed!==14)throw Error('Incomplete selected-neighbour coverage');
 return {cases:completed,compared,neighborEntries,nativeNormalizedExact:true,originalMarkerIdsPreserved:true,activityAndCompactionVerified:compact.nativeNormalizedExact,intermediateReadbackBytes:intermediateReadbackBytes+compact.intermediateReadbackBytes,cpuPhysics:false,neighborsConnected:true,fullSolverAdvanced:false};
}
