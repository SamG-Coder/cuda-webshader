import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try {
 const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto((process.env.CW_BASE_URL||`http://127.0.0.1:${server.address().port}`)+'/sandbox.html?example=bezier');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:120000});
 const report=await page.evaluate(async fixture=>{
  const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA GPU required');
  const expected=new Float32Array(Uint8Array.from(atob(fixture),c=>c.charCodeAt(0)).buffer),r=s.pipelineResult;
  if(r.geometryReadbackBytes!==0||r.controlReadbackBytes!==8)throw Error('Unexpected render-path readback');
  const records=await s.runtime.read(r.buffers.bLines,Uint32Array),index=r.pools.findIndex(t=>t.name==='device_vec2_f32_'),pool=await s.runtime.read(r.objectArena.buffers[index],Uint32Array),floats=new Float32Array(pool.buffer);
  let maxError=0,vertices=0;const handles=new Set();for(let i=0;i<256;i++){
   const handle=records[i*8+6],count=records[i*8+7];if(!handle||handles.has(handle)||pool[handle-1]!==count)throw Error('Invalid allocation');handles.add(handle);vertices+=count;
   for(let j=0;j<count;j++)for(let c=0;c<2;c++){const actual=floats[256+((handle-1)*32+j)*2+c],e=Math.abs(actual-expected[(i*32+j)*2+c]);if(!Number.isFinite(actual)||e>1e-6)throw Error('Native vertex mismatch');maxError=Math.max(maxError,e);}
  }
  if(vertices!==3958||!document.querySelector('#preview canvas')||r.draws<1)throw Error('Incomplete curve preview');
  return {passed:true,device:s.runtime.describe(),curves:256,vertices,maxError,geometryReadbackBytes:r.geometryReadbackBytes,controlReadbackBytes:r.controlReadbackBytes,softwareAdapterRequested:false};
 },(await readFile('reports/bezier-cdp-native.bin')).toString('base64'));
 if((await page.evaluate(()=>window.sandbox.editor.getValue())).replace(/\r\n/g,'\n')!==(await readFile('showcases/bezier/kernel.cu','utf8')).replace(/\r\n/g,'\n'))throw Error('CUDA source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==2)throw Error('Expected original parent and queued child shader');
 await page.locator('#shader-pass').selectOption('1');await page.screenshot({path:'reports/bezier-sandbox.png'});
 await page.locator('#preview canvas').screenshot({path:'showcases/bezier/preview.png'});
 const before=await page.evaluate(()=>({draws:window.sandbox.pipelineResult.draws,dispatches:window.sandbox.runtime.stats.dispatches}));await page.locator('#preview canvas').hover();await page.mouse.wheel(0,-250);
 await page.waitForFunction(before=>window.sandbox.pipelineResult.draws>before.draws,before);if(await page.evaluate(()=>window.sandbox.runtime.stats.dispatches)!==before.dispatches)throw Error('Camera movement reran compute');
 if(errors.length)throw Error(errors.join('\n'));
 await writeFile('reports/bezier-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true,zoomWithoutCompute:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
