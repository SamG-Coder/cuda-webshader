import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({headless:true,executablePath:process.env.CW_CHROMIUM||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{const page=await browser.newPage();await page.goto('http://localhost:5173/');const report=await page.evaluate(async()=>{
 const {GpuRuntime}=await import('/src/runtime/runtime.js'),runtime=await GpuRuntime.create(),results=[];
 try{const adapter=runtime.describe();if(!/nvidia/i.test(JSON.stringify(adapter)))throw Error('Real NVIDIA adapter required');
 for(const [index,entry] of [[18,'transposeCoalesced'],[19,'transposeNoBankConflicts'],[35,'transposeNaive']]){
  const source=await(await fetch('/showcases/nvidia/kernels/'+index+'.cu')).text(),kernel=await runtime.kernel(source,{entry,workgroupSize:[32,16,1]});
  for(const [width,height] of [[32,32],[64,96],[96,64],[128,128]]){
   const n=width*height,input=Float32Array.from({length:n+16},(_,i)=>i/8-width),initial=new Float32Array(n+16).fill(-12345),idata=runtime.createBuffer(input),odata=runtime.createBuffer(initial);
   try{runtime.batch().dispatch(kernel.bind({idata,odata},{width,height}),[width/32,height/32,1]).submit();const actual=await runtime.read(odata);let failures=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(actual[x*height+y]!==input[y*width+x])failures++;
    const guards=actual.slice(n).every(v=>v===-12345);results.push({entry,width,height,pass:failures===0&&guards,failures,guardValues:16,guardsPreserved:guards});
   }finally{runtime.destroyBuffer(idata);runtime.destroyBuffer(odata);}
  }
 }
 let divergentRejected=false;try{await runtime.kernel('__global__ void invalid(){cooperative_groups::thread_block b=cooperative_groups::this_thread_block();if(threadIdx.x==0u)cooperative_groups::sync(b);}',{workgroupSize:[4,1,1]});}catch(e){if(!/uniform/i.test(e.message))throw e;divergentRejected=true;}
 return {date:new Date().toISOString(),adapter,softwareAdapterRequested:false,sourceCompiledAtRuntime:true,results,divergentRejected,passed:results.every(r=>r.pass)&&divergentRejected};
 }finally{runtime.dispose();}
 });await writeFile('reports/nvidia-transpose.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));if(!report.passed)process.exitCode=1;
}finally{await browser.close();}
