import {chromium} from 'playwright';import {createStaticServer} from './serve.mjs';import {readFile,writeFile} from 'node:fs/promises';
const server=createStaticServer(new URL('../dist/',import.meta.url).pathname.replace(/^\/(\w:)/,'$1'));await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:'+server.address().port+'/sandbox.html?example=smoke');await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,null,{timeout:60000});
 if(await page.evaluate(()=>window.sandbox.lastError))throw Error(await page.evaluate(()=>window.sandbox.lastError));
 await page.locator('#stop').click();
 const report=await page.evaluate(async()=>{
  const s=window.sandbox,r=s.pipelineResult;await r.settle();if(s.runtime.describe().vendor!=='nvidia'||r.count!==16384||r.controlReadbackBytes!==0)throw Error('Wrong smoke device/count/readback');
  if(r.simulationSteps>64)throw Error('Missed native step');
  const before=s.runtime.stats.readbackBytes;
  while(r.simulationSteps<64)await r.stepFrame(r.simulationSteps*.5);
  if(s.runtime.stats.readbackBytes!==before)throw Error('Smoke feedback readback');
  const comparisons=[];
  for(const name of ['positions','velocities']){
   const actual=await s.runtime.read(r.buffers[name]),native=new Float32Array(await(await fetch('reports/smoke-showcase-64-'+name+'.bin')).arrayBuffer());let maxError=0;
   if(actual.length!==native.length)throw Error('Wrong native capture');for(let i=0;i<actual.length;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite smoke particle');maxError=Math.max(maxError,Math.abs(actual[i]-native[i]));}
   if(maxError>2e-6)throw Error('Native smoke mismatch '+name+': '+maxError);comparisons.push({name,maxError,components:actual.length});
  }
  window.smokePausedSteps=r.simulationSteps;window.smokeReadbackBytes=s.runtime.stats.readbackBytes;
  return {passed:true,particles:r.count,simulationSteps:r.simulationSteps,comparisons,device:s.runtime.describe(),intermediateReadbackBytes:0,softwareAdapterRequested:false};
 });
 const canvas=page.locator('#preview canvas'),box=await canvas.boundingBox();await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+box.width*.7,box.y+box.height*.6,{steps:12});await page.mouse.up();
 await page.evaluate(()=>window.sandbox.pipelineResult.settle());
 if(!await page.evaluate(()=>window.sandbox.pipelineResult.simulationSteps===window.smokePausedSteps&&window.sandbox.runtime.stats.readbackBytes===window.smokeReadbackBytes))throw Error('Paused orbit advanced simulation or read back data');
 if(await page.evaluate(()=>window.sandbox.editor.getValue())!==await readFile('showcases/smoke/kernel.cu','utf8'))throw Error('CUDA source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==6)throw Error('Expected six compare kernels');
 await page.screenshot({path:'reports/smoke-sandbox.png'});
 const previousFrames=await page.evaluate(()=>window.sandbox.pipelineResult.animationFrames);await page.locator('#animate').uncheck();await page.locator('#animate').check();await page.waitForFunction(n=>window.sandbox.pipelineResult.animationFrames>n,previousFrames);
 await page.setViewportSize({width:390,height:844});if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');
 await page.evaluate(()=>window.oldSmokeRenderer=window.sandbox.pipelineResult.smokeRenderer);const runs=await page.evaluate(()=>window.sandbox.completedRuns);await page.selectOption('#example','wave');await page.waitForFunction(n=>window.sandbox.completedRuns>n||window.sandbox.lastError,runs);if(!await page.evaluate(()=>window.oldSmokeRenderer.destroyed))throw Error('Smoke renderer not disposed');
 if(errors.length)throw Error(errors.join('\n'));
 await writeFile('reports/smoke-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true,pausedOrbit:true,pauseResume:true,mobileOverflow:false,rendererDisposed:true},null,2));console.log('PASS smoke sandbox: native step 64, orbit, animation, comparison, cleanup and mobile layout.');
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
