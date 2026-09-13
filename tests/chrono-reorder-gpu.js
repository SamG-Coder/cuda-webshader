import {decodeDiagnostics} from '../src/compiler/diagnostics.js';
export async function checkChronoReorder(runtime){
 const load=path=>fetch(new URL(path,import.meta.url));
 const source=await(await load('chrono-reorder.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),cases=await(await load('../reports/chrono-reorder-native.json')).json(),messages=await(await load('../reports/chrono-reorder-messages.json')).json();
 const input=await(await load('../reports/chrono-reorder-input.bin')).arrayBuffer(),native=await(await load('../reports/chrono-reorder-native.bin')).arrayBuffer();
 const kernel=await runtime.kernel(source,{entry:'reorderDataD',defines:{__CUDA_ARCH__:1},workgroupSize:[128]});let compared=0,captured=0;
 if(cases.length!==4||kernel.artifact.metadata.bindings.length!==16)throw Error('Incomplete reorder fixture or ABI');
 const inputNames=['posRadD','velMasD','rhoPresMuD','tauXxYyZzD','tauXyXzYzD','pcEvSvD','activityIdentifierOriginalD'],outputNames=['sortedPosRadD','sortedVelMasD','sortedRhoPreMuD','sortedTauXxYyZzD','sortedTauXyXzYzD','sortedPcEvSvD','activityIdentifierSortedD'],widths=[4,3,4,3,3,3,1];
 for(const c of cases){
  let at=c.inputOffset;const take=bytes=>{const result=runtime.createBuffer(new Uint8Array(input.slice(at,at+bytes)));at+=bytes;return result;};
  const buffers={gridMarkerIndexD:take(c.selected*4)};
  for(const [i,name]of inputNames.entries())buffers[name]=take(c.original*widths[i]*4);
  for(const [i,name]of outputNames.entries())buffers[name]=runtime.createBuffer(new (i===6?Int32Array:Float32Array)(c.selected*widths[i]).fill(42));
  const info=kernel.artifact.metadata.diagnostics;buffers[info.buffer]=runtime.createBuffer((2+info.capacity*info.strideWords)*4);
  try{
   runtime.batch().dispatch(kernel.bind(buffers,{...params,'constant.paramsD.physics_problem':c.mode%2,numActive:c.selected}),[Math.ceil(c.selected/128)]).submit();
   let offset=c.outputOffset;for(const [i,name]of outputNames.entries()){
    const actual=await runtime.read(buffers[name],Uint32Array),expected=new Uint32Array(native,offset,c.selected*widths[i]);if(actual.length!==expected.length)throw Error('Reorder output size changed');
    for(let j=0;j<actual.length;j++){if(actual[j]!==expected[j])throw Error(`Reorder case ${c.mode}, ${name}[${j}]: ${actual[j]} != ${expected[j]}`);compared++;}offset+=actual.byteLength;
   }
   const diagnostic=decodeDiagnostics(kernel.artifact.metadata,await runtime.read(buffers[info.buffer],Uint32Array));if(diagnostic.dropped||JSON.stringify(diagnostic.messages.sort())!==JSON.stringify([...messages[c.mode]].sort()))throw Error('Original reorder diagnostics differ from native CUDA');captured+=diagnostic.messages.length;
  }finally{for(const b of Object.values(buffers))runtime.destroyBuffer(b);}
 }
 const probe=await runtime.kernel('__global__ void diagnosticProbe(const float* input,unsigned int* finite){unsigned int i=threadIdx.x;finite[i]=isfinite(input[i]);printf("lane=%u signed=%d %%\\n",i,int(i)-2);}',{workgroupSize:[5],diagnosticCapacity:2});
 const info=probe.artifact.metadata.diagnostics,b={input:runtime.createBuffer(new Float32Array([0,Infinity,-Infinity,NaN,1])),finite:runtime.createBuffer(20),[info.buffer]:runtime.createBuffer((2+info.capacity*info.strideWords)*4)};
 try{runtime.batch().dispatch(probe.bind(b,{}),[1]).submit();const finite=[...await runtime.read(b.finite,Uint32Array)];if(JSON.stringify(finite)!=='[1,0,0,0,1]')throw Error('Finite classification mismatch');const d=decodeDiagnostics(probe.artifact.metadata,await runtime.read(b[info.buffer],Uint32Array));if(d.attempted!==5||d.dropped!==3||new Set(d.messages).size!==2||d.messages.some(s=>!Array.from({length:5},(_,i)=>`lane=${i} signed=${i-2} %\n`).includes(s)))throw Error('Bounded diagnostic capture mismatch');}finally{for(const v of Object.values(b))runtime.destroyBuffer(v);}
 return {cases:4,originalMarkers:521,selectedMarkers:259,compared,nativeExact:true,capturedNativeMessages:captured,storageBindings:16,cfdStressUntouched:true,crmStressReordered:true,nonfiniteDiagnosticsTested:true,diagnosticOverflowTested:true};
}
