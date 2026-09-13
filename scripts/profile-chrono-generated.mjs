// Diagnostic only: separates dispatches into timestamped passes. Its wall time
// is not comparable to production; use test-wgsl-trim for end-to-end timings.
import {createStaticServer} from './serve.mjs';import {chromium} from 'playwright';import {writeFileSync} from 'node:fs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const result=await page.evaluate(async()=>{
  const {GpuRuntime}=await import('/src/runtime/runtime.js'),{compile}=await import('/src/compiler/compiler.js'),{trimWgsl,tokens,pairs}=await import('/tests/experiments/wgsl-trim.js'),{optimizeFloatComparisons}=await import('/tests/experiments/wgsl-float-comparisons.js'),{createChronoLoop}=await import('/tests/chrono-loop-plan.js');
  const adapter=await navigator.gpu.requestAdapter();if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA required');
  const params=await(await fetch('/reports/chrono-params.json')).json(),initial={};for(const [key,path]of Object.entries({pos:'chrono-search-input.bin',vel:'chrono-marker-velocities.bin',rho:'chrono-marker-rhopremu.bin'}))initial[key]=new Uint8Array(await(await fetch('/reports/'+path)).arrayBuffer());
  const results=[];
  for(const mode of ['trimmed','comparisons']){
   const errors=[],runtime=await GpuRuntime.create({onError:e=>errors.push(e.message)}),original={},shaders=[],sets=[],records=[];let loop;
   try{
    if(!runtime.device.features.has('timestamp-query'))throw Error('GPU timestamps required');for(const [key,data]of Object.entries(initial))original[key]=runtime.createBuffer(data);
    const kernelFactory=async(source,options)=>{
     const a=compile(source,options),trim=trimWgsl(a),optimized=optimizeFloatComparisons(trim.artifact),selected=mode==='comparisons'?optimized.artifact:trim.artifact;
     const t=tokens(selected.wgsl),match=pairs(t),functions=[];
     for(let i=0;i<t.length;i++)if(t[i].value==='fn'){
      const name=t[i+1].value;let body=i+2;while(t[body].value!=='{')body++;const end=match.get(body),bodyTokens=t.slice(body+1,end);
      if(name==='main'||name.startsWith('f_'))functions.push({name,characters:t[end].end-t[i].start,loops:bodyTokens.filter(x=>['for','while','loop'].includes(x.value)).length,doubleCalls:bodyTokens.filter((x,j)=>/^cw_d_/.test(x.value)&&bodyTokens[j+1]?.value==='(').length,correctedDivisions:bodyTokens.filter(x=>x.value==='cw_divide_f32').length,storageReferences:bodyTokens.filter(x=>/^b_/.test(x.value)).length,aggregateCopies:bodyTokens.filter(x=>/^cw_value_copy_/.test(x.value)).length});i=end;
     }
     shaders.push({entry:options.entry,inputCharacters:source.length,originalCharacters:a.wgsl.length,trimmedCharacters:trim.artifact.wgsl.length,selectedCharacters:selected.wgsl.length,comparisonCandidates:optimized.replacements,functions});return runtime.kernel(selected);
    };
    loop=await createChronoLoop(runtime,params,30327,{kernelFactory});
    const originalBatch=runtime.batch.bind(runtime);
    runtime.batch=(options={})=>{
     const batch=originalBatch(options);let label=options.label??'compute';
     const dispatch=batch.dispatch.bind(batch);batch.dispatch=(invocation,groups,indirect)=>{batch.endPass();label=invocation.kernel.artifact.name;return dispatch(invocation,groups,indirect);};
     batch.beginPass=()=>{
      if(batch.pass)return batch.pass;
      const index=records.length,setIndex=Math.floor(index/2048),slot=(index%2048)*2;
      if(!sets[setIndex])sets[setIndex]=runtime.device.createQuerySet({type:'timestamp',count:4096});
      records.push({name:label,setIndex,slot});batch.pass=batch.encoder.beginComputePass({timestampWrites:{querySet:sets[setIndex],beginningOfPassWriteIndex:slot,endOfPassWriteIndex:slot+1}});batch.passCount++;batch.lastPipeline=null;batch.lastBindGroup=null;batch.lastOffset=-1;return batch.pass;
     };return batch;
    };
    for(let i=0;i<50;i++)await loop(original);await runtime.idle();
    const gpu={};for(let setIndex=0;setIndex<sets.length;setIndex++){
     const count=Math.min(2048,records.length-setIndex*2048),buffer=runtime.createBuffer(count*16,{usage:GPUBufferUsage.QUERY_RESOLVE}),encoder=runtime.device.createCommandEncoder();encoder.resolveQuerySet(sets[setIndex],0,count*2,buffer.gpuBuffer,0);runtime.device.queue.submit([encoder.finish()]);
     const words=await runtime.read(buffer,Uint32Array),times=new BigUint64Array(words.buffer);
     for(let j=0;j<count;j++){const name=records[setIndex*2048+j].name;gpu[name]??={milliseconds:0,dispatches:0};gpu[name].milliseconds+=Number(times[j*2+1]-times[j*2])/1e6;gpu[name].dispatches++;}runtime.destroyBuffer(buffer);
    }
    if(errors.length)throw Error(errors.join('\n'));results.push({mode,steps:50,shaders,gpu});
   }finally{loop?.dispose();for(const set of sets)set.destroy();runtime.dispose();}
  }
  return {device:'NVIDIA RTX 5080',softwareAdapterRequested:false,caveat:'Diagnostic per-dispatch passes; timestamp quantization and instrumentation affect measurements. Excludes copies/readbacks/host waits.',results};
 });writeFileSync('.local/chrono-generated-profile.json',JSON.stringify(result,null,2));console.log('Saved generated-code inventory and GPU profile');
}finally{await browser.close();await new Promise(r=>server.close(r));}
