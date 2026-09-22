import {chromium} from 'playwright';
import {createStaticServer} from './serve.mjs';
const server=createStaticServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
try {
 browser=await chromium.launch({...(process.env.CW_BROWSER?{channel:process.env.CW_BROWSER}:process.platform==='win32'?{channel:'msedge'}:{}),headless:true,args:['--enable-unsafe-webgpu']});
 const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/tests/least-squares-gpu.html`);
 await page.waitForFunction(()=>window.runLeastSquaresTests);
 console.log(JSON.stringify(await page.evaluate(()=>window.runLeastSquaresTests()),null,2));
}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}

