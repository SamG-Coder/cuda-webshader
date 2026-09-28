import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.env.CW_BEND_ROOT);await new Promise(r=>server.listen(0,'127.0.0.1',r));
let browser;
try {
  browser=await chromium.launch({headless:true,...(process.env.CW_CHROMIUM?{executablePath:process.env.CW_CHROMIUM}:{})});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  await page.goto(`http://127.0.0.1:${server.address().port}/bend.html`);
  const report=await page.evaluate(async()=>{
    const {compileBend}=await import('./src/bend/compiler.js');
    const {runBend}=await import('./src/bend/runtime.js');
    const {examples}=await import('./src/bend/examples.js');
    const {bendCases}=await import('./tests/bend-cases.js');
    const {GpuRuntime}=await import('./src/runtime/runtime.js');
    const source=await (await fetch('./src/bend/runtime.cu')).text();
    const runtime=await GpuRuntime.create(), cases=[];
    async function check(name,compiled,rows,want,options={}) {
      const result=await runBend(runtime,compiled,rows,options);
      if(result.status.some(Boolean)) throw Error(name+': '+JSON.stringify(result));
      if(result.output.some((x,i)=>Math.abs(x-want[i])>1e-6)) throw Error(name+': '+JSON.stringify(result.output)+' expected '+JSON.stringify(want));
      cases.push({name,rows:rows.length,output:result.output,elapsedMs:result.elapsedMs,maxHeapWords:Math.max(...result.usage.filter((_,i)=>i%2===0))});
    }
    try {
      for(const e of bendCases) {
        const c=compileBend(e.source,source,{entry:e.entry});
        await check(e.name,c,e.rows,e.expected);
      }
      const identity=compileBend(examples.proof.source,source,{entry:'identity'});
      for(const n of [1,63,64,65,257]) {const rows=Array.from({length:n},(_,i)=>[(i*2654435761)>>>0]);await check('dispatch-'+n,identity,rows,rows.map(x=>x[0]));}
      const recursive=compileBend(examples.pow2.source,source,{entry:'pow2'});
      const small=await runBend(runtime,recursive,[[0],[6],[1]],{arenaWords:512});
      if(small.status[0]!==0||small.status[1]!==1||small.status[2]!==0||small.output[0]!==1||small.output[2]!==2)throw Error('Arena exhaustion or lane isolation failed: '+JSON.stringify(small));
      const bounded=await runBend(runtime,recursive,[[6]],{maxSteps:1});
      if(bounded.status[0]!==2)throw Error('Step limit failed');
      const successor=compileBend(bendCases.find(x=>x.name==='nat-successor').source,source);
      const overflow=await runBend(runtime,successor,[[4294967295]]);
      if(overflow.status[0]!==5)throw Error('Nat overflow was not reported');
      const info=runtime.adapter?.info;
      return {passed:true,adapter:info?{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description}:null,cases,arenaIsolation:small.status,stepLimit:bounded.status,natOverflow:overflow.status,formalVerdict:false};
    } finally {runtime.dispose();}
  });
  await page.locator('#run').click();
  await page.waitForFunction(()=>window.__bendLastRun||window.__bendLastError,{},{timeout:30000});
  const ui=await page.evaluate(()=>({result:window.__bendLastRun,error:window.__bendLastError}));
  if(ui.error||ui.result.status.some(Boolean))throw Error('UI failed: '+JSON.stringify(ui));
  await mkdir(new URL('../reports/',import.meta.url),{recursive:true});
  await page.screenshot({path:new URL('../reports/bend-runtime.png',import.meta.url).pathname.replace(/^\/([A-Z]:)/,'$1'),fullPage:true});
  await writeFile(new URL('../reports/bend-runtime-gpu.json',import.meta.url),JSON.stringify({...report,ui:true},null,2));
  console.log(JSON.stringify({...report,cases:report.cases.map(({output,...c})=>c)},null,2));
} finally {await browser?.close();await new Promise(r=>server.close(r));}
