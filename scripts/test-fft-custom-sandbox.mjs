import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const variant=Number(process.env.CW_FFT_VARIANT||1),example=variant===1?'fft-custom':'fft-fused';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 const url=process.env.CW_PUBLIC_URL||`http://127.0.0.1:${server.address().port}/sandbox.html?example=${example}`;await page.goto(url);
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:60000});if(await page.evaluate(()=>window.sandbox.lastError))throw Error(await page.evaluate(()=>window.sandbox.lastError));
 const report=await page.evaluate(async variant=>{
  const s=window.sandbox,r=s.pipelineResult,native=new Float32Array(await(await fetch(`reports/fft-custom-full-${variant}-1-output.bin`)).arrayBuffer());
  if(s.runtime.describe().vendor!=='nvidia'||r.controlReadbackBytes!==0||r.inspection.length!==4194304)throw Error('Wrong adapter, readback or output size');
  let e2=0,r2=0,maxError=0;for(let i=0;i<native.length;i++){const d=r.inspection[i]-native[i];if(!Number.isFinite(d))throw Error('Nonfinite convolution');e2+=d*d;r2+=native[i]**2;maxError=Math.max(maxError,Math.abs(d));}const relativeL2=Math.sqrt(e2/r2);if(relativeL2>1e-6)throw Error('Native convolution mismatch');
  const pixels=document.querySelector('#preview canvas').getContext('2d').getImageData(0,0,2048,2048).data,levels=new Set();for(let i=0;i<pixels.length;i+=4)levels.add(pixels[i]);if(levels.size<30)throw Error('Convolution preview lost scalar contrast');
  return {passed:true,relativeL2,maxError,components:native.length,grayscaleLevels:levels.size,device:s.runtime.describe(),intermediateReadbackBytes:0,softwareAdapterRequested:false};
 },variant);
 const source=await readFile('showcases/fft-convolution-custom/kernel.cu','utf8');if((await page.evaluate(()=>window.sandbox.editor.getValue())).replaceAll('\r\n','\n')!==source.replaceAll('\r\n','\n'))throw Error('Original source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();for(const name of ['padKernel_kernel','padDataClampToBorder_kernel',...(variant===1?['spPostprocess2D_kernel','spPreprocess2D_kernel','modulateAndNormalize_kernel']:['spProcess2D_kernel']),'forwardFftAxis','inverseFftAxis'])if(!passes.some(p=>p.includes(name)))throw Error('Missing shader '+name);
 await page.screenshot({path:'.local/'+example+'-sandbox.png'});await page.setViewportSize({width:390,height:844});if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');
 const before=await page.evaluate(()=>window.sandbox.completedRuns);await page.selectOption('#example','wave');await page.waitForFunction(n=>window.sandbox.completedRuns>n||window.sandbox.lastError,before);if(await page.evaluate(()=>window.sandbox.lastError)||errors.length)throw Error(errors.join('\n')||'Preset switching failed');
 await writeFile('reports/'+example+'-sandbox-check.json',JSON.stringify({...report,sourceUnchanged:true,passes,mobileOverflow:false,presetSwitching:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
