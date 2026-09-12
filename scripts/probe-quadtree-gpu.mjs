import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
// Validate the complete imported kernel without claiming recursive execution.
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
 browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage();await page.goto('http://127.0.0.1:'+server.address().port+'/');
 const report=await page.evaluate(async()=>{
  const {compile}=await import('/src/compiler/compiler.js');
  const source=(await Promise.all(['quadtree-cdp-device.cuh','quadtree-cdp-setup.cuh'].map(async f=>(await fetch('/tests/'+f)).text()))).join('\n');
  const artifact=compile(source,{entry:'build_quadtree_kernel<128>',valueBuffers:['nodes','points'],workgroupSize:[128],sharedMemoryBytes:64,objectHeap:'persistent',deviceHeap:{maxAllocations:1024,maxElements:1024},deviceLaunchQueue:{maxLaunches:1024}});
  const adapter=await navigator.gpu.requestAdapter();if(!adapter||adapter.info.isFallbackAdapter||adapter.info.vendor!=='nvidia')throw Error('This quadtree probe requires a real NVIDIA adapter.');
  const device=await adapter.requestDevice({requiredFeatures:artifact.metadata.requiredFeatures});
  try{
   const info=await device.createShaderModule({code:artifact.wgsl}).getCompilationInfo();
   return {mode:'Real NVIDIA shader validation; no dispatch',device:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},compiled:true,validated:!info.messages.some(m=>m.type==='error'),executed:false,softwareAdapterRequested:false,messages:info.messages.map(m=>({type:m.type,line:m.lineNum,message:m.message}))};
  }finally{device.destroy();}
 });
 await writeFile('reports/quadtree-gpu-probe.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
