// Opt-in experiment only: npm/build/sandbox do not enable this pass.
import {createStaticServer} from './serve.mjs';
import {chromium} from 'playwright';
import {writeFileSync} from 'node:fs';
const steps=Number(process.argv[2]??3000),output=process.argv[3]??'.local/wgsl-trim-benchmark.json';
if(!Number.isInteger(steps)||steps<1||steps>10000)throw Error('Steps must be 1..10000');
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage();page.on('console',m=>console.log(m.text()));await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const result=await page.evaluate(async steps=>{
  const {GpuRuntime}=await import('/src/runtime/runtime.js'),{compile}=await import('/src/compiler/compiler.js'),{trimWgsl}=await import('/tests/experiments/wgsl-trim.js'),{createChronoLoop}=await import('/tests/chrono-loop-plan.js');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA adapter required');
  const params=await(await fetch('/reports/chrono-params.json')).json(),initial={};
  for(const [name,path]of Object.entries({pos:'chrono-search-input.bin',vel:'chrono-marker-velocities.bin',rho:'chrono-marker-rhopremu.bin'}))initial[name]=new Uint8Array(await(await fetch('/reports/'+path)).arrayBuffer());
  const runs=[];let reference=null;
  for(const mode of ['original','trimmed','trimmed','original']){
   console.log('Preparing '+mode+' run '+(runs.length+1));
   const errors=[],runtime=await GpuRuntime.create({onError:e=>errors.push(e.message)}),original={},shaders=[];let loop;
   try{
    for(const [name,bytes]of Object.entries(initial))original[name]=runtime.createBuffer(bytes);
    const kernelFactory=async(source,options)=>{
     let time=performance.now();const artifact=compile(source,options),compileMs=performance.now()-time;
     time=performance.now();const trimmed=mode==='trimmed'?trimWgsl(artifact):null,trimMs=performance.now()-time;
     time=performance.now();const kernel=await runtime.kernel(trimmed?.artifact??artifact),pipelineMs=performance.now()-time;
     shaders.push({entry:options.entry,compileMs,trimMs,pipelineMs,...(trimmed?.stats??{beforeBytes:artifact.wgsl.length,afterBytes:artifact.wgsl.length,projections:0,removedFunctions:0})});return kernel;
    };
    loop=await createChronoLoop(runtime,params,30327,{kernelFactory});
    const counts=[],start=performance.now();for(let i=0;i<steps;i++){counts.push((await loop(original)).neighbors);}
    await runtime.idle();const elapsedMs=performance.now()-start,values={};
    for(const [name,buffer]of Object.entries(original))values[name]=Array.from(await runtime.read(buffer,Uint32Array));
    if(errors.length)throw Error(errors.join('\n'));
    if(!reference)reference={counts,values};
    const exactCounts=counts.every((v,i)=>v===reference.counts[i]),exactBits=Object.keys(values).every(key=>values[key].every((v,i)=>v===reference.values[key][i]));
    if(!exactCounts||!exactBits)throw Error('Trim changed particle bits or neighbour counts');
    const run={mode,steps,elapsedMs,exactCounts,exactBits,shaders};runs.push(run);console.log(JSON.stringify({mode,elapsedMs,exactCounts,exactBits}));
   }finally{loop?.dispose();runtime.dispose();}
  }
  return {device:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},softwareAdapterRequested:false,order:'ABBA',steps,timing:'Loop plus final GPU completion; compilation and output inspection excluded',runs};
 },steps);
 writeFileSync(output,JSON.stringify(result,null,2));console.log('Saved '+output);
}finally{await browser.close();await new Promise(r=>server.close(r));}
