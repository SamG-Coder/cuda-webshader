import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1100}});
 await page.goto((process.env.CW_BASE_URL||`http://127.0.0.1:${server.address().port}`)+'/sandbox.html?example=cubemap');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:60000});
 const report=await page.evaluate(async()=>{const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');const r=s.pipelineResult,actual=await s.runtime.read(r.buffers.g_odata),atlas=await s.runtime.read(r.buffers.image),expected=new Float32Array(await(await fetch('reports/cubemap-native.bin')).arrayBuffer());if(actual.length!==24576)throw Error('Wrong cube size');for(let i=0;i<actual.length;i++)if(actual[i]!==expected[i])throw Error('Native face mismatch '+i);const canvas=document.querySelector('#preview .image-output canvas');if(!canvas||canvas.width!==192||canvas.height!==128)throw Error('Missing six-face atlas');const pixels=canvas.getContext('2d').getImageData(0,0,192,128).data;for(let y=0;y<128;y++)for(let x=0;x<192;x++){const i=y*192+x,face=Math.floor(y/64)*3+Math.floor(x/64),value=expected[face*4096+(y%64)*64+x%64];if(atlas[i]!==value)throw Error('Atlas orientation mismatch');const shade=Math.round(Math.max(0,Math.min(1,(value+24575)/24575))*255);for(let c=0;c<4;c++)if(pixels[i*4+c]!== (c===3?255:shade))throw Error('Preview pixel mismatch '+i);}return {passed:true,device:s.runtime.describe(),faces:6,width:64,values:24576,nativeValuesExact:true,previewPixelsVerified:24576,controlReadbackBytes:r.controlReadbackBytes,softwareAdapterRequested:false};});
 if((await page.evaluate(()=>window.sandbox.editor.getValue())).replace(/\r\n/g,'\n')!==(await readFile('showcases/cubemap/kernel.cu','utf8')).replace(/\r\n/g,'\n'))throw Error('Source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==2)throw Error('Missing shader comparison');
 await page.screenshot({path:'reports/cubemap-sandbox.png'});await writeFile('reports/cubemap-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
