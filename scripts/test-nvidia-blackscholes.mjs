import {chromium} from 'playwright';import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{const page=await browser.newPage();await page.goto('http://localhost:5173/');const report=await page.evaluate(async()=>{
 const {GpuRuntime}=await import('/src/runtime/runtime.js'),{blackScholesFixture}=await import('/showcases/nvidia/blackscholes-fixtures.js'),runtime=await GpuRuntime.create(),results=[];
 try{const adapter=runtime.describe();if(!/nvidia/i.test(JSON.stringify(adapter)))throw Error('Real NVIDIA adapter required');const source=await(await fetch('/showcases/nvidia/kernels/21.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[128,1,1]});
 for(const optN of [0,2,10,258,259,1024]){const f=blackScholesFixture(optN,16),resources=Object.fromEntries(Object.entries(f.buffers).map(([name,data])=>[name,runtime.createBuffer(data)]));
 try{runtime.batch().dispatch(kernel.bind(resources,f.scalars),f.groups).submit();const outputs={};for(const [name,expected]of Object.entries(f.expectedOutputs)){const values=await runtime.read(resources[name]),maxError=Math.max(...values.map((v,i)=>Math.abs(v-expected[i]))),guardsPreserved=values.slice(Math.floor(optN/2)*2).every(v=>v===-12345);outputs[name]={maxError,guardsPreserved,pass:guardsPreserved&&values.every((v,i)=>Number.isFinite(v)&&Math.abs(v-expected[i])<=f.absoluteTolerance+f.relativeTolerance*Math.abs(expected[i]))};}results.push({optN,outputs,pass:Object.values(outputs).every(o=>o.pass)});
 }finally{await runtime.idle();for(const r of Object.values(resources))runtime.destroyBuffer(r);}}
 return {date:new Date().toISOString(),adapter,softwareAdapterRequested:false,sourceCompiled:true,absoluteTolerance:0.0002,relativeTolerance:0.00002,results,passed:results.every(r=>r.pass)};
 }finally{runtime.dispose();}
 });await writeFile('reports/nvidia-blackscholes.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}finally{await browser.close();}
