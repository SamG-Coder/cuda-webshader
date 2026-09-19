import {chromium} from 'playwright';
import {writeFile,mkdir} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
let browser;
try{
 const software=process.env.CW_SOFTWARE_GPU==='1';
 browser=await chromium.launch({headless:true,...(process.env.CW_CHROMIUM?{executablePath:process.env.CW_CHROMIUM}:{}),
  args:software?['--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-unsafe-swiftshader']:[]});
 const page=await browser.newPage();
 await page.goto(`http://127.0.0.1:${server.address().port}/tests/dependency-optimization-gpu.html`);
 await page.waitForFunction(()=>window.__optimizerReport!==undefined,{},{timeout:120000});
 const report=await page.evaluate(()=>window.__optimizerReport);
 report.softwareAdapterRequested=software;report.browser=browser.version();
 await mkdir('reports',{recursive:true});
 await writeFile('reports/dependency-optimization-gpu.json',JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report,null,2));
 if(!report.passed)process.exitCode=1;
 // Module validation is separate from pipeline creation and output equivalence.
 // Validate all 11 real Stratum entries, in all three compiler modes.
 if(process.argv.includes('--stratum')){
  const validation=await page.evaluate(async()=>{
   const a=await navigator.gpu.requestAdapter(),d=await a.requestDevice(),rows=[];
   try{
    for(const entry of ['initCamera','stepCamera','clearQueue','prepareLots','planBounds','buildGroupBounds','reduceGroupBounds','tracePrimary','reflectPixels','shadePixels','resolveFrame'])for(const mode of ['none','dependencies','specialize']){
     const r=await fetch(`/reports/optimizer-wgsl/${entry}-${mode}.wgsl`);if(!r.ok)throw Error('Missing WGSL fixture '+entry);
     const info=await d.createShaderModule({code:await r.text()}).getCompilationInfo();
     rows.push({entry,mode,errors:info.messages.filter(m=>m.type==='error').map(m=>({message:m.message,lineNum:m.lineNum,linePos:m.linePos}))});
    }
   }finally{d.destroy();}
   return rows;
  });
  await writeFile('reports/dependency-optimization-wgsl-validation.json',JSON.stringify(validation,null,2)+'\n');
  console.log('Stratum modules validated:',validation.length,'errors:',validation.reduce((n,v)=>n+v.errors.length,0));
  if(validation.some(v=>v.errors.length))process.exitCode=1;
 }
}finally{await browser?.close();await new Promise(r=>server.close(r));}
