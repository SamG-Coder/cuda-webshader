export async function checkChronoSearch(runtime){
 const load=name=>fetch(new URL(name,import.meta.url));
 const source=await(await load('chrono-search.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),input=new Float32Array(await(await load('../reports/chrono-search-input.bin')).arrayBuffer()),reference=await(await load('../reports/chrono-search-native.json')).json(),native=new Uint32Array(await(await load('../reports/chrono-search-native.bin')).arrayBuffer());
 const n=input.length/4,cells=params['constant.paramsD.gridSize.x']*params['constant.paramsD.gridSize.y']*params['constant.paramsD.gridSize.z'];
 if(n!==30327||reference.markers!==n||reference.fluid!==16731||reference.cells!==cells||reference.sections.reduce((a,b)=>a+b,0)!==native.length)throw Error('Incomplete native neighbour reference');
 const kernels={};for(const entry of ['hashMarkers','gatherMarkers','findCellStartEndD','neighborSearchNum','neighborSearchID'])kernels[entry]=await runtime.kernel(source,{entry,workgroupSize:[128],...(entry==='findCellStartEndD'?{sharedMemoryBytes:516,predicatedReturns:true}:{})});
 const buffers={positions:runtime.createBuffer(input),hashes:runtime.createBuffer(n*4),indices:runtime.createBuffer(n*4),sortedPosRad:runtime.createBuffer(n*16),sortedRhoPreMu:runtime.createBuffer(16),cellStartD:runtime.createBuffer(cells*4),cellEndD:runtime.createBuffer(cells*4),counts:runtime.createBuffer((n+1)*4),offsets:runtime.createBuffer((n+1)*4),total:runtime.createBuffer(4)};
 const dispatch=(entry,args)=>{const k=kernels[entry],values=Object.fromEntries(k.artifact.metadata.scalars.map(p=>[p.name,p.origin==='constant'?params[p.name]:n]));runtime.batch().dispatch(k.bind(args,values),[Math.ceil(n/128)]).submit();};
 let neighbors,compared=0;
 try{
  const before={...runtime.stats};
  dispatch('hashMarkers',{positions:buffers.positions,hashes:buffers.hashes,indices:buffers.indices});await runtime.sortPairs(buffers.hashes,buffers.indices,{count:n});
  dispatch('gatherMarkers',{positions:buffers.positions,indices:buffers.indices,sortedPosRad:buffers.sortedPosRad});
  dispatch('findCellStartEndD',{cellStartD:buffers.cellStartD,cellEndD:buffers.cellEndD,gridMarkerHashD:buffers.hashes,gridMarkerIndexD:buffers.indices});
  const search={sortedPosRad:buffers.sortedPosRad,sortedRhoPreMu:buffers.sortedRhoPreMu,cellStart:buffers.cellStartD,cellEnd:buffers.cellEndD};
  dispatch('neighborSearchNum',{...search,numNeighborsPerPart:buffers.counts});await runtime.exclusiveScan(buffers.counts,buffers.offsets,{count:n+1,total:buffers.total});
  const total=(await runtime.read(buffers.total,Uint32Array))[0];if(total!==reference.neighbors)throw Error('GPU neighbour allocation total differs from native CUDA: '+total);
  // Match the native host allocation after the GPU count/scan stages.
  neighbors=runtime.createBuffer(total*4);dispatch('neighborSearchID',{...search,numNeighborsPerPart:buffers.offsets,neighborList:neighbors});await runtime.idle();
  const intermediateReadback=runtime.stats.readbackBytes-before.readbackBytes;if(intermediateReadback!==4||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected host transfer during neighbour construction');
  let offset=0;for(const [i,buffer] of [buffers.hashes,buffers.indices,buffers.cellStartD,buffers.cellEndD,buffers.counts,buffers.offsets,neighbors].entries()){
   const actual=await runtime.read(buffer,Uint32Array);if(actual.length!==reference.sections[i])throw Error('Native section size mismatch '+i);
   for(let j=0;j<actual.length;j++){if(actual[j]!==native[offset+j])throw Error(`Chrono neighbour section ${i}, index ${j}: ${actual[j]} != ${native[offset+j]}`);compared++;}offset+=actual.length;
  }
  return {markers:n,fluidMarkers:16731,boundaryMarkers:n-16731,neighborEntries:total,compared,nativeExact:true,intermediateReadbackBytes:intermediateReadback,cpuPhysics:false};
 }finally{if(neighbors)runtime.destroyBuffer(neighbors);for(const b of Object.values(buffers))runtime.destroyBuffer(b);}
}
