import {chromium} from 'playwright';import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{const page=await browser.newPage();await page.goto('http://localhost:5173/');const report=await page.evaluate(async()=>{
 const {GpuRuntime}=await import('/src/runtime/runtime.js'),{scalarCases,scalarFixture,mul24Inputs,mul24Source,mul24Expected}=await import('/showcases/nvidia/scalar-fixtures.js'),runtime=await GpuRuntime.create(),results=[];
 try{const adapter=runtime.describe();if(!/nvidia/i.test(JSON.stringify(adapter)))throw Error('Real NVIDIA adapter required');const source=await(await fetch('/showcases/nvidia/kernels/20.cu')).text();
 for(const test of scalarCases){const f=scalarFixture(test),resources=Object.fromEntries(Object.entries(f.buffers).map(([name,data])=>[name,runtime.createBuffer(data)]));
  try{const kernel=await runtime.kernel(source,{workgroupSize:[test.threads,1,1]});runtime.batch().dispatch(kernel.bind(resources,f.scalars),f.groups).submit();const values=await runtime.read(resources.d_C);results.push({...test,pass:values.every((v,i)=>v===f.expected[i]),guardsPreserved:values.slice(test.vectorN).every(v=>v===-12345),valuesChecked:values.length});}finally{for(const r of Object.values(resources))runtime.destroyBuffer(r);}
 }
 const pairs=mul24Inputs.flatMap(a=>mul24Inputs.map(b=>[a,b])),n=pairs.length,resources={a:runtime.createBuffer(Int32Array.from(pairs,p=>p[0])),b:runtime.createBuffer(Int32Array.from(pairs,p=>p[1])),signedOut:runtime.createBuffer(new Int32Array(n+16).fill(-12345)),unsignedOut:runtime.createBuffer(new Uint32Array(n+16).fill(12345))};
 let integerChecks,literalChecks;
 try{const kernel=await runtime.kernel(mul24Source,{workgroupSize:[128,1,1]});runtime.batch().dispatch(kernel.bind(resources,{n}),[1,1,1]).submit();const signed=await runtime.read(resources.signedOut,Int32Array),unsigned=await runtime.read(resources.unsignedOut,Uint32Array);integerChecks={pairs:n,pass:pairs.every(([a,b],i)=>signed[i]===mul24Expected(a,b,true)&&unsigned[i]===mul24Expected(a,b,false))&&signed.slice(n).every(v=>v===-12345)&&unsigned.slice(n).every(v=>v===12345)};
 const literals=await runtime.kernel('__global__ void literals(int* s,unsigned int* u){s[0]=__mul24(8388607,8388607);u[0]=__umul24(16777215u,16777215u);}',{workgroupSize:[1,1,1]});runtime.batch().dispatch(literals.bind({s:resources.signedOut,u:resources.unsignedOut},{}),[1,1,1]).submit();literalChecks=(await runtime.read(resources.signedOut,Int32Array))[0]===mul24Expected(8388607,8388607,true)&&(await runtime.read(resources.unsignedOut,Uint32Array))[0]===mul24Expected(16777215,16777215,false);
 }finally{for(const r of Object.values(resources))runtime.destroyBuffer(r);}
 return {date:new Date().toISOString(),adapter,softwareAdapterRequested:false,results,integerChecks,overflowingLiteralsPassed:literalChecks,passed:results.every(r=>r.pass)&&integerChecks.pass&&literalChecks};
 }finally{runtime.dispose();}
 });await writeFile('reports/nvidia-scalar.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}finally{await browser.close();}
