import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1100}});
 await page.goto((process.env.CW_BASE_URL||`http://127.0.0.1:${server.address().port}`)+'/sandbox.html?example=binomial');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:60000});
 const report=await page.evaluate(async()=>{const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');const r=s.pipelineResult,actual=await s.runtime.read(r.buffers.d_CallValue),expected=new Float32Array(await(await fetch('reports/binomial-options-native.bin')).arrayBuffer());let delta=0,reference=0,maxError=0;for(let i=0;i<1024;i++){const e=Math.abs(actual[i]-expected[i]);if(!Number.isFinite(actual[i])||e>2e-4+Math.abs(expected[i])*5e-4)throw Error('Native result mismatch '+i);delta+=e;reference+=Math.abs(expected[i]);maxError=Math.max(maxError,e);}if(delta/reference>5e-4)throw Error('Native L1 mismatch');const canvas=document.querySelector('#preview .image-output canvas');if(!canvas||canvas.width!==32||canvas.height!==32)throw Error('Missing heatmap');const pixels=canvas.getContext('2d').getImageData(0,0,32,32).data;for(let i=0;i<1024;i++){const shade=Math.round(Math.max(0,Math.min(1,actual[i]/30))*255);for(let c=0;c<4;c++)if(pixels[i*4+c]!== (c===3?255:shade))throw Error('Preview pixel mismatch '+i);}return {passed:true,device:s.runtime.describe(),options:1024,steps:2048,previewPixelsVerified:1024,l1:delta/reference,maxAbsoluteError:maxError,controlReadbackBytes:r.controlReadbackBytes,softwareAdapterRequested:false};});
 if((await page.evaluate(()=>window.sandbox.editor.getValue())).replace(/\r\n/g,'\n')!==(await readFile('showcases/binomial-options/kernel.cu','utf8')).replace(/\r\n/g,'\n'))throw Error('Source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==1)throw Error('Missing shader comparison');
 await page.screenshot({path:'reports/binomial-sandbox.png'});await writeFile('reports/binomial-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
