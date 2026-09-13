import {checkChronoActivity} from './chrono-activity-gpu.js';
export async function checkChronoCompact(runtime,{afterCompact}={}){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('chrono-compact.cu')).text(),cases=await(await load('../reports/chrono-compact-native.json')).json(),native=await(await load('../reports/chrono-compact-native.bin')).arrayBuffer();
 const kernels={};for(const entry of ['normalizeActivity','fillActiveListD','gatherSelected'])kernels[entry]=await runtime.kernel(source,{entry,workgroupSize:[128]});
 let compared=0,intermediateReadbackBytes=0,completed=0;
 const activity=await checkChronoActivity(runtime,{afterDispatch:async({caseInfo:c,buffers})=>{
  const ref=cases[c.mode],n=c.n;if(!ref||ref.n!==n||ref.mode!==c.mode)throw Error('Missing compaction reference');
  const positive=runtime.createBuffer(n*4),prefix=runtime.createBuffer(n*4),list=runtime.createBuffer(new Uint32Array(n).fill(0xffffffff)),total=runtime.createBuffer(4);let selected;
  try{
   const before={...runtime.stats};
   runtime.batch().dispatch(kernels.normalizeActivity.bind({activity:buffers.extendedActivityIdD,positive},{n}),[Math.ceil(n/128)]).submit();
   await runtime.exclusiveScan(positive,prefix,{count:n,total});
   runtime.batch().dispatch(kernels.fillActiveListD.bind({prefixSum:prefix,extendedActivityIdD:buffers.extendedActivityIdD,activeListD:list},{numAllMarkers:n}),[Math.ceil(n/128)]).submit();
   const count=(await runtime.read(total,Uint32Array))[0];if(count!==ref.total)throw Error('Normalized native/GPU selected count differs');
   selected=runtime.createBuffer(Math.max(count,1)*16);
   if(count)runtime.batch().dispatch(kernels.gatherSelected.bind({positions:buffers.posRadD,activeList:list,selected},{n:count}),[Math.ceil(count/128)]).submit();
   await runtime.idle();
   const readback=runtime.stats.readbackBytes-before.readbackBytes;if(readback!==4||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Unexpected CPU transfer in activity compaction');intermediateReadbackBytes+=readback;
   if(afterCompact)await afterCompact({caseInfo:c,buffers,activeList:list,count});
   let offset=ref.offset+n*4; // Original faulty native scan is retained separately, never substituted as the corrected reference.
   for(const [buffer,words] of [[prefix,n],[list,n],[selected,count*4]]){
    if(!words)continue;const actual=await runtime.read(buffer,Uint32Array),expected=new Uint32Array(native,offset,words);
    for(let i=0;i<words;i++){if(actual[i]!==expected[i])throw Error(`Normalized compaction case ${c.mode}, word ${i}: ${actual[i]} != ${expected[i]}`);compared++;}offset+=words*4;
   }
   completed++;
  }finally{for(const b of [positive,prefix,list,total,selected])if(b)runtime.destroyBuffer(b);}
 }});
 if(completed!==14)throw Error('Incomplete activity/compaction coverage');
 return {cases:completed,compared,nativeNormalizedExact:true,upstreamScanMismatchCases:cases.filter(c=>c.baselinePrefixMismatches).length,upstreamPrefixMismatches:cases.reduce((n,c)=>n+c.baselinePrefixMismatches,0),upstreamWriteCollisions:cases.reduce((n,c)=>n+c.baselineWriteCollisions,0),explicitPositiveFlagNormalization:true,activityKernelNativeExact:activity.nativeExact,intermediateReadbackBytes,cpuPhysics:false,neighborsConnected:false};
}
