import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(new URL('../dist/',import.meta.url).pathname.replace(/^\/(\w:)/,'$1'));await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`http://127.0.0.1:${server.address().port}/sandbox.html?example=marching`);
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:60000});
 const result=await page.evaluate(async()=>{
  const s=window.sandbox;if(s.lastError)throw Error(s.lastError);const r=s.pipelineResult;if(s.runtime.describe().vendor!=='nvidia')throw Error('NVIDIA hardware required');
  const comparisons=[];for(const [name,resource]of [['positions','pos'],['normals','norm']]){const actual=await s.runtime.read(r.buffers[resource],Float32Array,r.count*16),expected=new Float32Array(await(await fetch('/reports/marching-triangles-native-'+name+'.bin')).arrayBuffer());if(actual.length!==expected.length)throw Error('Native capture size mismatch');let maxError=0;for(let i=0;i<actual.length;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite '+name);maxError=Math.max(maxError,Math.abs(actual[i]-expected[i]));}if(maxError>1e-6)throw Error(name+' mismatch '+maxError);comparisons.push({name,components:actual.length,maxError});}
  if(r.count!==6240||r.controlReadbackBytes!==8)throw Error('Unexpected vertex count or control transfers');return {device:s.runtime.describe(),vertices:r.count,triangles:r.count/3,controlReadbackBytes:r.controlReadbackBytes,comparisons};
 });
 const source=await page.evaluate(()=>window.sandbox.editor.getValue());if(source!==await readFile('showcases/marching-cubes/kernel.cu','utf8'))throw Error('CUDA source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==5)throw Error('Missing generated compute passes: '+passes);
 for(let i=0;i<passes.length;i++){await page.selectOption('#shader-pass',String(i));if(!await page.locator('#wgsl-column').isVisible())throw Error('WGSL comparison hidden');}
 await page.selectOption('#shader-pass','4');
 const before=await page.evaluate(()=>({...window.sandbox.runtime.stats}));const canvas=page.locator('#preview canvas'),box=await canvas.boundingBox();const imageBefore=await canvas.screenshot();
 await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width/2+110,box.y+box.height/2+35,{steps:15});await page.mouse.up();
 const imageAfter=await canvas.screenshot();if(imageBefore.equals(imageAfter))throw Error('Orbit did not change rendered mesh');
 const after=await page.evaluate(()=>({...window.sandbox.runtime.stats}));for(const key of ['dispatches','dataBytesUploaded','readbackBytes'])if(before[key]!==after[key])throw Error('Orbit transferred or recomputed geometry');
 await page.screenshot({path:'reports/marching-cubes-sandbox.png'});
 await page.locator('#settings summary').click();const config=JSON.parse(await page.locator('#config').inputValue());for(const step of config.pipeline.steps)if(step.scalars&&'isoValue'in step.scalars)step.scalars.isoValue=-1000;
 await page.locator('#config').fill(JSON.stringify(config));await page.locator('#run').click();await page.waitForFunction(()=>window.sandbox.completedRuns===2||window.sandbox.lastError,{},{timeout:60000});
 if(await page.evaluate(()=>window.sandbox.lastError||window.sandbox.pipelineResult.count!==0))throw Error('Empty surface failed');
 if(await page.evaluate(()=>window.sandbox.editor.getValue())!==source)throw Error('Host settings modified CUDA');
 await page.setViewportSize({width:390,height:844});if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');if(errors.length)throw Error(errors.join('\n'));
 await writeFile('reports/marching-cubes-sandbox-check.json',JSON.stringify({passed:true,staticBuild:true,softwareAdapterRequested:false,...result,passes,sourceUnchanged:true,orbitWithoutComputeOrDataTransfer:true,emptySurfacePassed:true,mobileOverflow:false},null,2));
 console.log('PASS marching-cubes sandbox',JSON.stringify(result));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
