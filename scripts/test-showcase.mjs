import {chromium} from 'playwright';
import {writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const source=await readFile('showcases/simplegl/kernel.cu','utf8'),origin=JSON.parse(await readFile('showcases/simplegl/upstream.json','utf8'));
const kernel=source.match(/__global__ void simple_vbo_kernel\([\s\S]*?\n\}/)[0];
if(createHash('sha256').update(kernel).digest('hex')!==origin.kernelSha256)throw new Error('Upstream kernel changed');
const browser=await chromium.launch({headless:true,...(process.env.CW_CHROMIUM?{executablePath:process.env.CW_CHROMIUM}:{})});
try{const page=await browser.newPage({viewport:{width:1440,height:960}}),errors=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(`${process.env.CW_BASE_URL||'http://localhost:5173'}/showcases/simplegl/`);await page.waitForFunction(()=>window.showcase?.report||window.showcaseError);if(await page.evaluate(()=>window.showcaseError))throw new Error(await page.evaluate(()=>window.showcaseError));const results=[];
for(const size of [128,256,512,1024]){await page.locator('#size').selectOption(String(size));await page.waitForFunction(n=>window.showcase.size===n,size);results.push(await page.evaluate(()=>window.showcase.verify()));}
await page.locator('#size').selectOption('256');await page.waitForFunction(()=>window.showcase.size===256);await page.waitForFunction(()=>document.querySelector('#fps').textContent.includes('65536'));
await page.screenshot({path:'reports/simplegl-showcase.png'});await page.locator('#pause').click();const t=await page.evaluate(()=>window.showcase.time);await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));if(await page.evaluate(()=>window.showcase.time)!==t)throw new Error('Pause failed');if(errors.length)throw new Error(errors.join('\n'));
const report={passed:true,unmodifiedUpstreamKernel:true,numericalCases:results.length*3,results,pause:'PASS',pageErrors:errors};await writeFile('reports/simplegl-webgpu.json',JSON.stringify(report,null,2));console.log(`PASS ${report.numericalCases} real WebGPU cases, original kernel hash, rendering and pause`);
}finally{await browser.close();}
