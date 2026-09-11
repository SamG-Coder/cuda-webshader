import {chromium} from 'playwright';import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{const page=await browser.newPage();await page.goto('http://localhost:5173/');const report=await page.evaluate(async()=>{
 const {GpuRuntime}=await import('/src/runtime/runtime.js'),{fwtSharedFixture}=await import('/showcases/nvidia/fwt-shared-fixtures.js'),runtime=await GpuRuntime.create(),results=[];
 try{const adapter=runtime.describe();if(!/nvidia/i.test(JSON.stringify(adapter)))throw Error('Real NVIDIA adapter required');const source=await(await fetch('/showcases/nvidia/kernels/30.cu')).text();
 for(const log2N of [2,3,6,7,8,11]){const N=2**log2N,f=fwtSharedFixture(log2N,3,16),resources=Object.fromEntries(Object.entries(f.buffers).map(([name,data])=>[name,runtime.createBuffer(data)]));try{const k=await runtime.kernel(source,{workgroupSize:[N/4,1,1],sharedMemoryBytes:N*4});runtime.batch().dispatch(k.bind(resources,f.scalars),f.groups).submit();const out=await runtime.read(resources.d_Output);results.push({log2N,N,batches:3,threads:N/4,sharedMemoryBytes:N*4,guardsPreserved:out.slice(-16).every(v=>v===-12345),pass:out.every((v,i)=>v===f.expected[i])});}finally{await runtime.idle();for(const r of Object.values(resources))runtime.destroyBuffer(r);}}
 return {date:new Date().toISOString(),adapter,softwareAdapterRequested:false,sourceCompiled:true,reference:'Direct Walsh matrix using parity of row & column, independent of staged butterflies',results,passed:results.every(r=>r.pass)};
 }finally{runtime.dispose();}
 });await writeFile('reports/nvidia-fwt-shared.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}finally{await browser.close();}
