import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{const page=await browser.newPage();await page.goto('http://localhost:5173/showcases/nvidia/');const report=await page.evaluate(async()=>{const {checkAll}=await import('./check.js');return checkAll();});await writeFile('reports/nvidia-gpu.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(report.results.some(r=>!r.pass))process.exitCode=1;}finally{await browser.close();}
