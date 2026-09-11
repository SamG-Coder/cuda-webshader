import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createStaticServer} from './serve.mjs';
const browser=await chromium.launch({headless:true,...(process.env.CW_CHROMIUM?{executablePath:process.env.CW_CHROMIUM}:{})}),server=createStaticServer(fileURLToPath(new URL('../dist/',import.meta.url)));
await new Promise(r=>server.listen(0,'127.0.0.1',r));const page=await browser.newPage({viewport:{width:1440,height:960}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 const results=[];
 for(const [name,url] of [['source','http://127.0.0.1:5173'],['dist',`http://127.0.0.1:${server.address().port}`]]){
  await page.goto(url);await page.waitForFunction(()=>window.cudaWebShader?.demo&&document.querySelector('#simulation-state').textContent==='RUNNING');
  await page.waitForFunction(()=>document.querySelector('#hud-fps').textContent.match(/[1-9]/));
  const state=await page.evaluate(()=>({device:document.querySelector('#device-status').textContent,particles:document.querySelector('#hud-count').textContent,cadence:document.querySelector('#hud-fps').textContent,fatalVisible:!document.querySelector('#fatal').hidden,overflow:document.documentElement.scrollWidth>innerWidth}));
  if(state.fatalVisible||state.overflow)throw new Error(JSON.stringify(state));
  if(name==='source')await page.screenshot({path:'reports/live-app.png'});
  await page.locator('#pause').click();await page.waitForFunction(()=>document.querySelector('#simulation-state').textContent==='PAUSED');
  results.push({name,...state,pause:'PASS'});
 }
 await page.goto('http://127.0.0.1:5173/reports/performance-comparison.html');
 if(await page.locator('tbody tr:visible').count()!==10)throw new Error('Large report rows missing');
 await page.screenshot({path:'reports/performance-comparison.png',fullPage:true});
 await page.locator('#scale').selectOption('small');if(await page.locator('tbody tr:visible').count()!==10)throw new Error('Small report rows missing');
 if(errors.length)throw new Error(errors.join('\n'));
 await writeFile('reports/app-smoke.json',JSON.stringify({passed:true,results,reportFilter:'PASS',pageErrors:errors},null,2));console.log(JSON.stringify(results,null,2));
}finally{await browser.close();await new Promise(r=>server.close(r));}
