import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer(process.cwd()+'/dist');await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:1500,height:1100}});
 await page.goto((process.env.CW_BASE_URL||`http://127.0.0.1:${server.address().port}`)+'/sandbox.html?example=dxt');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,{},{timeout:60000});
 const report=await page.evaluate(async()=>{
  const s=window.sandbox;if(s.lastError)throw Error(s.lastError);if(s.runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');
  const r=s.pipelineResult,compressed=await s.runtime.read(r.buffers.compressed,Uint32Array),pixels=await s.runtime.read(r.buffers.pixels,Uint32Array);
  const native=new Uint32Array(await(await fetch('reports/dxt-native.dds')).arrayBuffer(),128);
  if(compressed.length!==32768||native.length!==compressed.length||pixels.length!==262144)throw Error('Wrong image size');
  for(let i=0;i<compressed.length;i++)if(compressed[i]!==native[i])throw Error('Native word mismatch '+i);
  const rgb=v=>{const a=[v>>11&31,v>>5&63,v&31];return [(a[0]<<3)|(a[0]>>2),(a[1]<<2)|(a[1]>>4),(a[2]<<3)|(a[2]>>2),255];};
  let nonblack=0;
  for(let y=0;y<512;y++)for(let x=0;x<512;x++){
   const b=(Math.floor(y/4)*128+Math.floor(x/4))*2,a=native[b]&65535,z=native[b]>>>16,c0=rgb(a),c1=rgb(z);
   const palette=[c0,c1,a>z?c0.map((v,i)=>i===3?255:Math.floor((2*v+c1[i])/3)):c0.map((v,i)=>i===3?255:Math.floor((v+c1[i])/2)),a>z?c0.map((v,i)=>i===3?255:Math.floor((v+2*c1[i])/3)):[0,0,0,0]];
   const c=palette[(native[b+1]>>>(2*((y%4)*4+x%4)))&3],expected=(c[0]|c[1]<<8|c[2]<<16|c[3]<<24)>>>0;
   if(pixels[y*512+x]!==expected)throw Error('Decoded pixel mismatch '+x+','+y);if(expected&16777215)nonblack++;
  }
  if(nonblack<10000)throw Error('Empty decoded image');return {passed:true,device:s.runtime.describe(),blocks:16384,nativeWordsMatched:32768,decodedPixelsMatched:262144,nonblack,softwareAdapterRequested:false};
 });
 if((await page.evaluate(()=>window.sandbox.editor.getValue())).replace(/\r\n/g,'\n')!==(await readFile('showcases/dxt/kernel.cu','utf8')).replace(/\r\n/g,'\n'))throw Error('Source changed');
 await page.locator('#tab-compare').click();const passes=await page.locator('#shader-pass option').allTextContents();if(passes.length!==2)throw Error('Missing shader comparison');
 await page.screenshot({path:'reports/dxt-sandbox.png'});await writeFile('reports/dxt-sandbox-check.json',JSON.stringify({...report,passes,sourceUnchanged:true},null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
