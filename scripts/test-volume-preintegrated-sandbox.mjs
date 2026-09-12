import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(fileURLToPath(new URL('../dist/',import.meta.url)));await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`http://127.0.0.1:${server.address().port}/sandbox.html?example=volume-preintegrated`);
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,null,{timeout:60000});
 const report=await page.evaluate(async()=>{
  const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Expected NVIDIA hardware');
  const result=s.pipelineResult;window.__previousVolumeTable=result.buffers.tables;if(result.controlReadbackBytes!==0)throw Error('Intermediate CPU readback');
  const actual=new Uint8Array(result.inspection.buffer),expected=new Uint8Array(await(await fetch('reports/volume-preintegrated-1024-0-native.bin')).arrayBuffer());
  let maxChannelError=0,coloredPixels=0;for(let i=0;i<actual.length;i++)maxChannelError=Math.max(maxChannelError,Math.abs(actual[i]-expected[i]));
  for(let i=0;i<actual.length;i+=4)if(Math.max(...actual.subarray(i,i+3))-Math.min(...actual.subarray(i,i+3))>20)coloredPixels++;
  if(maxChannelError>4||coloredPixels<1000)throw Error('Native image or colour check failed');
  const source=await(await fetch('showcases/volume-preintegrated/kernel.cu')).text();if(s.editor.getValue().replaceAll('\r\n','\n')!==source.replaceAll('\r\n','\n'))throw Error('Source editor changed');
  return {passed:true,maxChannelError,coloredPixels,controlReadbackBytes:result.controlReadbackBytes,textureLayers:result.buffers.tables.depth,tableWidth:result.buffers.tables.width,adapter:s.runtime.describe(),softwareAdapterRequested:false};
 });
 await page.locator('#tab-compare').click();if(!await page.locator('#wgsl-column').isVisible())throw Error('WGSL comparison unavailable');
 await page.locator('#preview canvas').screenshot({path:'reports/volume-preintegrated-preview.png'});
 await page.screenshot({path:'reports/volume-preintegrated-sandbox.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');
 await page.selectOption('#example','wave');await page.waitForFunction(()=>window.sandbox.completedRuns>=2||window.sandbox.lastError);if(await page.evaluate(()=>window.sandbox.lastError))throw Error('Switching showcase failed');
 if(!await page.evaluate(()=>window.__previousVolumeTable.destroyed))throw Error('Layered texture was not freed on example switch');
 if(errors.length)throw Error(errors.join('\n'));await writeFile('reports/volume-preintegrated-sandbox-check.json',JSON.stringify(report,null,2)+'\n');console.log(report);
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
