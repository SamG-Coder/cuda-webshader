import {chromium} from 'playwright';import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{const page=await browser.newPage();await page.goto('http://localhost:5173/');const report=await page.evaluate(async()=>{
 const {GpuRuntime}=await import('/src/runtime/runtime.js'),{matrixFixture}=await import('/showcases/nvidia/matrixmul-fixtures.js'),runtime=await GpuRuntime.create(),results=[];
 try{const adapter=runtime.describe();if(!/nvidia/i.test(JSON.stringify(adapter)))throw Error('Real NVIDIA adapter required');const source=await(await fetch('/showcases/nvidia/kernels/22.cu')).text();
 for(const tile of [16,32]){const kernel=await runtime.kernel(source,{entry:`MatrixMulCUDA<${tile}>`,workgroupSize:[tile,tile,1]});for(const [M,N,K] of [[tile,tile,tile],[tile,2*tile,3*tile],[3*tile,tile,2*tile],[64,96,128]]){
 const f=matrixFixture(tile,M,N,K,16),resources=Object.fromEntries(Object.entries(f.buffers).map(([name,data])=>[name,runtime.createBuffer(data)]));
 try{runtime.batch().dispatch(kernel.bind(resources,f.scalars),f.groups).submit();const output=await runtime.read(resources.C),guardsPreserved=output.slice(M*N).every(v=>v===-12345);results.push({tile,M,N,K,guardsPreserved,valuesChecked:output.length,maxError:Math.max(...output.map((v,i)=>Math.abs(v-f.expected[i]))),pass:guardsPreserved&&output.every((v,i)=>v===f.expected[i])});}
 finally{await runtime.idle();for(const r of Object.values(resources))runtime.destroyBuffer(r);}}
 }
 return {date:new Date().toISOString(),adapter,sourceCompiled:true,softwareAdapterRequested:false,results,passed:results.every(r=>r.pass)};
 }finally{runtime.dispose();}
 });await writeFile('reports/nvidia-matrixmul.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}finally{await browser.close();}
