import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
 browser=await chromium.launch({headless:true,...(process.env.CW_CHROMIUM?{executablePath:process.env.CW_CHROMIUM}:{})});
 const page=await browser.newPage();page.on('pageerror',e=>console.error(e));
 await page.goto(`http://127.0.0.1:${server.address().port}/benchmarks/`);
 await page.waitForFunction(()=>window.comparisonReport!==undefined,{},{timeout:600000});
 const report=await page.evaluate(()=>window.comparisonReport);
 await writeFile(new URL('../reports/comparison-webgpu.json',import.meta.url),JSON.stringify(report,null,2));
 if(report.error)throw new Error(report.error);
 for(const r of report.results)console.log(`${r.key}: ${(r.gpuMedianMs*1000).toFixed(2)} us GPU / ${(r.wallMedianMs*1000).toFixed(2)} us wall`);
}finally{await browser?.close();await new Promise(r=>server.close(r));}
