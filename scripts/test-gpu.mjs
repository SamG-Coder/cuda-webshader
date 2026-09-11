import {chromium} from 'playwright';import {writeFile,mkdir} from 'node:fs/promises';import {createStaticServer} from './serve.mjs';
const server=createStaticServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const port=server.address().port;let browser;
await mkdir(new URL('../reports/',import.meta.url),{recursive:true});
try{
 const software=process.env.CW_SOFTWARE_GPU==='1',args=software?['--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-unsafe-swiftshader']:[];
 browser=await chromium.launch({headless:process.env.CW_HEADED!=='1',...(process.env.CW_CHROMIUM?{executablePath:process.env.CW_CHROMIUM}:{}),args});
 const page=await browser.newPage();page.on('console',m=>{if(m.type()==='error')console.error(m.text());});
 await page.goto(`http://127.0.0.1:${port}/tests/gpu.html${process.env.CW_BENCH==='1'?'?bench=1':''}`);
 await page.waitForFunction(()=>window.__gpuReport!==undefined,{},{timeout:180000});const report=await page.evaluate(()=>window.__gpuReport);report.softwareAdapterRequested=software;
 await writeFile(new URL('../reports/gpu-local.json',import.meta.url),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.failed)process.exitCode=1;
}catch(error){const report={mode:'NOT RUN',error:String(error.stack||error)};await writeFile(new URL('../reports/gpu-local.json',import.meta.url),JSON.stringify(report,null,2));console.error(report.error);process.exitCode=1;}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
