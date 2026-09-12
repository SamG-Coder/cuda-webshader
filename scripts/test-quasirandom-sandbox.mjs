import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1100}});
 await page.goto((process.env.CW_BASE_URL||`http://127.0.0.1:${server.address().port}`)+'/sandbox.html?example=quasirandom');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:60000});
 const report=await page.evaluate(async()=>{
 const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');const r=s.pipelineResult,values=await s.runtime.read(r.buffers.values,Uint32Array),points=await s.runtime.read(r.buffers.positions,Uint32Array),native=new Uint32Array(await(await fetch('reports/quasirandom-0-native.bin')).arrayBuffer()),N=1048576;
 if(values.length!==3*N||native.length!==3*N||points.length!==4*N)throw Error('Wrong point count');for(let i=0;i<3*N;i++)if(values[i]!==native[i])throw Error('Native mismatch '+i);
 for(let i=0;i<N;i++){for(let d=0;d<3;d++)if(points[4*i+d]!==native[d*N+i])throw Error('Point packing mismatch '+i);if(points[4*i+3]!==1065353216)throw Error('Invalid homogeneous coordinate');}
 return {passed:true,device:s.runtime.describe(),points:N,matchedNativeCoordinates:3*N,sourceCoordinatesUnchanged:true,controlReadbackBytes:r.controlReadbackBytes,softwareAdapterRequested:false};
 });
 if((await page.evaluate(()=>window.sandbox.editor.getValue())).replace(/\r\n/g,'\n')!==(await readFile('showcases/quasirandom/kernel.cu','utf8')).replace(/\r\n/g,'\n'))throw Error('Source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==2)throw Error('Missing shader comparison');
 await page.screenshot({path:'reports/quasirandom-sandbox.png'});await writeFile('reports/quasirandom-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
