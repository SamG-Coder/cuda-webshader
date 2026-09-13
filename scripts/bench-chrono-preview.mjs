import {createStaticServer} from './serve.mjs';
import {chromium} from 'playwright';
import {writeFileSync} from 'node:fs';
const server=createStaticServer();
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try {
 const page=await browser.newPage({viewport:{width:1600,height:1000}});
 await page.goto(`http://127.0.0.1:${server.address().port}/sandbox.html?example=chrono`);
 await page.waitForFunction(()=>window.sandbox?.pipelineResult?.steps>=25||window.sandbox?.lastError,{},{timeout:240000});
 const result=await page.evaluate(async()=>{
  const s=window.sandbox;
  if(s.lastError)throw Error(s.lastError);
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA required');
  const start=s.pipelineResult.steps,t=performance.now();
  await new Promise(r=>setTimeout(r,10000));
  const elapsedMs=performance.now()-t,steps=s.pipelineResult.steps-start;
  document.querySelector('#stop').click();await s.pipelineResult.settle();
  const stopped=s.pipelineResult.steps;await new Promise(r=>setTimeout(r,100));
  if(s.pipelineResult.steps!==stopped)throw Error('Simulation continued after stop');
  if(s.lastError)throw Error(s.lastError);
  return {steps,elapsedMs,stepsPerSecond:steps*1000/elapsedMs,simulatedSecondsPerWallSecond:steps*.0001*1000/elapsedMs,stopped,device:adapter.info.vendor,viewport:[1600,1000]};
 });
 writeFileSync(process.argv[2]??'.local/chrono-preview-bench.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{await browser.close();await new Promise(r=>server.close(r));}
