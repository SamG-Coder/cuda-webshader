import {createStaticServer} from './serve.mjs';
import {chromium} from 'playwright';
import {writeFileSync} from 'node:fs';
const steps=Number(process.argv[2]??1000);
if(!Number.isInteger(steps)||steps<1||steps>100000)throw Error('Invalid step count');
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const result=await page.evaluate(async steps=>{
  const {GpuRuntime}=await import('/src/runtime/runtime.js');
  const {createChronoLoop}=await import('/tests/chrono-loop-plan.js');
  const errors=[],runtime=await GpuRuntime.create({onError:e=>errors.push(e.message)});
  let loop;
  try{
   const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
   if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA required');
   const params=await(await fetch('/reports/chrono-params.json')).json(),original={};
   for(const [name,path]of Object.entries({pos:'chrono-search-input.bin',vel:'chrono-marker-velocities.bin',rho:'chrono-marker-rhopremu.bin'}))original[name]=runtime.createBuffer(new Uint8Array(await(await fetch('/reports/'+path)).arrayBuffer()));
   loop=await createChronoLoop(runtime,params,30327);
   const counts=[],start=performance.now();
   for(let i=0;i<steps;i++){const result=await loop(original);counts.push(result.neighbors);}
   const values={};for(const [name,buffer]of Object.entries(original))values[name]=Array.from(await runtime.read(buffer));
   if(errors.length)throw Error(errors.join('\n'));
   return {steps,elapsedMs:performance.now()-start,counts,values,stats:runtime.stats};
  }finally{loop?.dispose();runtime.dispose();}
 },steps);
 writeFileSync(process.argv[3]??'.local/chrono-loop-benchmark.json',JSON.stringify(result));console.log(JSON.stringify({steps:result.steps,elapsedMs:result.elapsedMs}));
}finally{await browser.close();await new Promise(r=>server.close(r));}
