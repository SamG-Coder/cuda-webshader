import {chromium} from 'playwright';import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{const page=await browser.newPage();await page.goto('http://localhost:5173/');const report=await page.evaluate(async()=>{
 const {GpuRuntime}=await import('/src/runtime/runtime.js'),{alignedCopyFixture}=await import('/showcases/nvidia/aligned-copy-fixtures.js'),runtime=await GpuRuntime.create(),results=[];
 try{const adapter=runtime.describe();if(!/nvidia/i.test(JSON.stringify(adapter)))throw Error('Real NVIDIA adapter required');const source=await(await fetch('/showcases/nvidia/kernels/26.cu')).text();
 for(const type of ['int','uint4','float4']){const kernel=await runtime.kernel(source,{entry:`testKernel<${type}>`,workgroupSize:[128,1,1]});for(const n of [0,1,129,1031]){const f=alignedCopyFixture(type,n,2,128,16),resources=Object.fromEntries(Object.entries(f.buffers).map(([name,data])=>[name,runtime.createBuffer(data)]));try{runtime.batch().dispatch(kernel.bind(resources,f.scalars),f.groups).submit();const output=await runtime.read(resources.d_odata,f.expected.constructor),actualBytes=new Uint8Array(output.buffer),expectedBytes=new Uint8Array(f.expected.buffer);results.push({type,n,bytesChecked:actualBytes.length,pass:actualBytes.every((v,i)=>v===expectedBytes[i])});}finally{await runtime.idle();for(const r of Object.values(resources))runtime.destroyBuffer(r);}}}
 return {date:new Date().toISOString(),adapter,softwareAdapterRequested:false,sourceCompiled:true,comparison:'byte-for-byte, including guards and signed zero',results,passed:results.every(r=>r.pass)};
 }finally{runtime.dispose();}
 });await writeFile('reports/nvidia-aligned-copy.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}finally{await browser.close();}
