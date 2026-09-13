export async function checkChronoActivity(runtime,{afterDispatch}={}){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('chrono-activity.cu')).text(),params=await(await load('../reports/chrono-params.json')).json();
 const cases=await(await load('../reports/chrono-activity-native.json')).json();
 const input=await(await load('../reports/chrono-activity-input.bin')).arrayBuffer(),reference=await(await load('../reports/chrono-activity-native.bin')).arrayBuffer();
 if(cases.length!==14||cases[0].n!==30327)throw Error('Incomplete native activity reference');
 const kernel=await runtime.kernel(source,{entry:'UpdateActivityD',workgroupSize:[128]});
 for(const name of ['ad_body_D','ad_node1D_D','ad_node2D_D'])if(kernel.artifact.metadata.bindings.find(b=>b.name===name).stride!==52)throw Error('Native ActiveDomain stride changed');
 let compared=0;const states=new Set();
 for(const c of cases){
  const n=c.n;let at=c.inputOffset;
  const take=bytes=>{const b=runtime.createBuffer(new Uint8Array(input.slice(at,at+bytes)));at+=bytes;return b;};
  const buffers={posRadD:take(n*16),rhoPreMuD:take(n*16),velMasD:take(n*12),ad_body_D:take(104),ad_node1D_D:take(104),ad_node2D_D:take(104),pos_bodies_D:runtime.createBuffer(12),pos_nodes1D_D:runtime.createBuffer(12),pos_nodes2D_D:runtime.createBuffer(12),activityIdentifierD:runtime.createBuffer(new Int32Array(n).fill(77)),extendedActivityIdD:runtime.createBuffer(new Int32Array(n).fill(77))};
  const scalars={...params,has_ad:!!c.has_ad,'time.lo':c['time.lo'],'time.hi':c['time.hi'],'constant.countersD.numAllMarkers.lo':n,'constant.countersD.numFluidMarkers.lo':c.mode===0?16731:99};
  for(const name of ['numFsiBodies','numFsiNodes1D','numFsiNodes2D'])scalars['constant.countersD.'+name+'.lo']=c.domainCount;
  if(c.mode)for(const axis of 'xyz'){scalars['constant.paramsD.worldOrigin.'+axis]=-1;scalars['constant.paramsD.boxDims.'+axis]=2;}
  if(c.mode)scalars['constant.paramsD.free_flow_duration']=.5;
  for(const [i,axis] of [...'xyz'].entries())scalars['constant.paramsD.'+axis+'_periodic']=!!c.periodic[i];
  try{
   runtime.batch().dispatch(kernel.bind(buffers,scalars),[Math.ceil(n/128)]).submit();
   if(afterDispatch)await afterDispatch({caseInfo:c,buffers});
   let offset=c.outputOffset;
   for(const name of ['activityIdentifierD','extendedActivityIdD','velMasD']){
    const actual=await runtime.read(buffers[name],Uint32Array),expected=new Uint32Array(reference,offset,actual.length);
    for(let i=0;i<actual.length;i++){if(actual[i]!==expected[i])throw Error(`Activity case ${c.mode}, ${name}[${i}]: ${actual[i]} != ${expected[i]}`);if(name==='activityIdentifierD')states.add(actual[i]|0);compared++;}
    offset+=actual.byteLength;
   }
  }finally{for(const buffer of Object.values(buffers))runtime.destroyBuffer(buffer);}
 }
 if(![-1,0,1].every(s=>states.has(s)))throw Error('Activity validation missed a marker state');
 const abiSource=await(await load('native-records.cu')).text(),abiInput=await(await load('../reports/native-records-input.bin')).arrayBuffer(),abiExpected=await(await load('../reports/native-records-native.json')).json();
 const abi=await runtime.kernel(abiSource,{entry:'nativeRecords',workgroupSize:[1]}),abiBuffers={data:runtime.createBuffer(new Uint8Array(abiInput)),output:runtime.createBuffer(16),flags:runtime.createBuffer(new Int32Array([0,1,2,3,4]))};
 try{runtime.batch().dispatch(abi.bind(abiBuffers,{}),[1]).submit();const actual=[...await runtime.read(abiBuffers.output,Float32Array),...await runtime.read(abiBuffers.flags,Int32Array)];if(JSON.stringify(actual)!==JSON.stringify(abiExpected))throw Error('Native adjacent bool / storage reference ABI mismatch');}finally{for(const b of Object.values(abiBuffers))runtime.destroyBuffer(b);}
 return {originalKernel:'UpdateActivityD',initializedMarkerPositions:30327,controlledCases:13,compared,nativeExact:true,nativeDomainStride:52,nonzeroPaddingTested:true,additionalNativeAbiValues:9,activityStates:[...states].sort(),physicsStepAdvanced:false};
}
