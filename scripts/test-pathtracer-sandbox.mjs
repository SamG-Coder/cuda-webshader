import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1600,height:1000}});
 await page.goto((process.env.CW_BASE_URL||`http://127.0.0.1:${server.address().port}`)+'/sandbox.html?example=pathtracer');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:120000});
 const native=await readFile('reports/pathtracer-full-native.bin');
 const report=await page.evaluate(async(base64)=>{
  const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA adapter required');
  const bytes=Uint8Array.from(atob(base64),c=>c.charCodeAt(0)),expected=new Float32Array(bytes.buffer),actual=await s.runtime.read(s.pipelineResult.buffers.fb);
  if(actual.length!==1200*800*3||expected.length!==actual.length)throw Error('Framebuffer size mismatch');
  let sum=0,max=0,close=0;for(let i=0;i<actual.length;i++){if(!Number.isFinite(actual[i]))throw Error('Non-finite output');const e=Math.abs(actual[i]-expected[i]);sum+=e;max=Math.max(max,e);if(e<1e-4)close++;}
  if(sum/actual.length>.002||close/actual.length<.95)throw Error('Native comparison outside stochastic tolerance');
  const canvas=document.querySelector('#preview .image-output canvas');if(!canvas||canvas.width!==1200||canvas.height!==800)throw Error('Image preview missing');
  const pixels=canvas.getContext('2d').getImageData(0,0,1200,800).data;
  for(let y=0;y<800;y++)for(let x=0;x<1200;x++)for(let c=0;c<3;c++){const expectedByte=Math.floor(Math.max(0,Math.min(.999,actual[((799-y)*1200+x)*3+c]))*255.99);if(pixels[(y*1200+x)*4+c]!==expectedByte)throw Error('Preview does not match kernel output');}
  return {passed:true,device:s.runtime.describe(),width:1200,height:800,samplesPerPixel:10,spheres:488,components:actual.length,meanAbsoluteError:sum/actual.length,maxAbsoluteError:max,fractionWithin1eMinus4:close/actual.length,previewPixelsVerified:960000,softwareAdapterRequested:false};
 },native.toString('base64'));
 if((await page.evaluate(()=>window.sandbox.editor.getValue())).replace(/\r\n/g,'\n')!==(await readFile('showcases/pathtracer/kernel.cu','utf8')).replace(/\r\n/g,'\n'))throw Error('CUDA source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==5)throw Error('Expected five generated compute shaders');
 await page.locator('#shader-pass').selectOption('3');
 await page.screenshot({path:'reports/pathtracer-sandbox.png'});
 await page.locator('#tab-cuda').click();await page.screenshot({path:'reports/pathtracer-sandbox-source.png'});
 const persistentState=await page.evaluate(async(source)=>{
  const r=window.sandbox.runtime,arena=r.createObjectArena(),objects=r.createBuffer(8),out=r.createBuffer(16),kernels={};
  try{for(const entry of ['create','update','cleanup'])kernels[entry]=await r.kernel(source,{entry,objectHeap:'persistent',workgroupSize:[1,1,1]});
   for(const entry of ['create','update','update','cleanup']){r.batch().dispatch(kernels[entry].bind({objects,...(entry==='create'?{}:{out})},{},{objectArena:arena}),[1,1,1]).submit();await r.idle();}
   const values=[...await r.read(out,Int32Array)];if(JSON.stringify(values)!=='[16,35,16,35]')throw Error('Persistent reference mutation or indexed assignment ordering failed: '+values);
   if((await r.read(objects,Uint32Array)).some(Boolean))throw Error('Object handles not cleared');
   for(const b of arena.buffers)if((await r.read(b,Uint32Array)).slice(0,1024).some(Boolean))throw Error('Persistent member test leaked objects');
   return {values,separateSubmissions:true,allFreed:true};
  }finally{arena.dispose();r.destroyBuffer(objects);r.destroyBuffer(out);}
 },await readFile('tests/persistent-member-state.cu','utf8'));
 report.persistentMemberState=persistentState;
 await writeFile('reports/pathtracer-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
