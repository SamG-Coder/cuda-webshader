import {chromium} from 'playwright';
import {fileURLToPath} from 'node:url';
import {createStaticServer} from './serve.mjs';

const server=createStaticServer(fileURLToPath(new URL('../dist/',import.meta.url)));
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try {
  const page=await browser.newPage(),requests=[];
  page.on('request',request=>{
    const url=new URL(request.url());
    if(url.pathname.startsWith('/src/')&&url.pathname.endsWith('.js')) requests.push(url);
  });
  // A stale cached module must never be requested by the new build.
  await page.route('**/src/**/*.js',route=>route.fulfill({contentType:'text/javascript',body:'throw Error("Stale unversioned module loaded");'}));
  await page.goto(`http://127.0.0.1:${server.address().port}/sandbox.html?example=particle-collision`);
  await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,null,{timeout:60000});
  const result=await page.evaluate(()=>({error:window.sandbox.lastError,vendor:window.sandbox.runtime?.describe().vendor}));
  if(result.error) throw Error(result.error);
  if(result.vendor!=='nvidia') throw Error('Expected real NVIDIA adapter');
  if(!requests.some(url=>url.pathname.endsWith('/worker.js'))||requests.some(url=>!url.searchParams.has('v'))) throw Error('Unversioned project module or missing compiler worker');
  console.log(`PASS: ${requests.length} versioned project module requests; old modules bypassed; particle showcase runs on NVIDIA.`);
} finally {
  await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
}
