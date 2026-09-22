import {chromium} from 'playwright';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
  browser=await chromium.launch({headless:true,...(process.env.CW_CHROMIUM?{executablePath:process.env.CW_CHROMIUM}:{}),args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/tests/half.test.mjs`);
  const result=await page.evaluate(async()=>{const {GpuRuntime}=await import('/src/runtime/runtime.js'),{checkHalf}=await import('/tests/half-gpu.js');const runtime=await GpuRuntime.create();try{return await checkHalf(runtime);}finally{runtime.dispose();}});
  console.log(JSON.stringify(result));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
