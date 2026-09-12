import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1100}});
 await page.goto((process.env.CW_BASE_URL||`http://127.0.0.1:${server.address().port}`)+'/sandbox.html?example=odd-even-sort');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:60000});
 const report=await page.evaluate(async()=>{const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');const r=s.pipelineResult,keys=await s.runtime.read(r.buffers.keys,Uint32Array),values=await s.runtime.read(r.buffers.values,Uint32Array),image=await s.runtime.read(r.buffers.image,Uint32Array),input=new Uint32Array(await(await fetch('reports/odd-even-input-keys.bin')).arrayBuffer()),manifest=await(await fetch('reports/odd-even-native-hashes.json')).json(),ref=manifest.cases.find(c=>c.length===1048576&&c.direction===0),hash=async words=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',words.buffer)),b=>b.toString(16).padStart(2,'0')).join('');if(keys.length!==1048576||values.length!==1048576||image.length!==2097152||await hash(keys)!==ref.keys||await hash(values)!==ref.values)throw Error('Native result mismatch');for(let i=0;i<image.length;i++){const x=i%2048,row=Math.floor(i/2048),expected=x<1024?input[row*1024+x]:keys[row*1024+x-1024];if(image[i]!==expected)throw Error('Preview mismatch');}return {passed:true,device:s.runtime.describe(),elements:1048576,keysAndValuesMatchNative:true,previewPixelsVerified:2097152,controlReadbackBytes:r.controlReadbackBytes,softwareAdapterRequested:false};});
 if((await page.evaluate(()=>window.sandbox.editor.getValue())).replace(/\r\n/g,'\n')!==(await readFile('showcases/odd-even-sort/kernel.cu','utf8')).replace(/\r\n/g,'\n'))throw Error('Source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==4)throw Error('Missing shader comparison');
 await page.screenshot({path:'reports/odd-even-sandbox.png'});await writeFile('reports/odd-even-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
