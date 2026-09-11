import {prepareReduction} from '../src/runtime/operations.js';import {kernelOptions} from '../src/kernels.js';
import {makeCases,compareArrays} from './cases.js';
export async function runGpuSuite(runtime,sources,{onCase=()=>{}}={}){
  const results=[],start=performance.now();
  const run=async(name,action)=>{const t=performance.now();let result;try{result={name,pass:true,...await action()};}catch(error){result={name,pass:false,error:String(error.stack||error)};}result.wallMs=performance.now()-t;results.push(result);onCase(result,results.length);};
  for(const c of makeCases())await run(c.name,async()=>{
    const buffers={};try{
      for(const [name,data]of Object.entries(c.buffers))buffers[name]=runtime.createBuffer(data,{label:`${c.name}: ${name}`});
      const kernel=await runtime.kernel(sources[c.id],kernelOptions(c.id));
      const invocation=kernel.bind(buffers,c.scalars);runtime.batch().dispatch(invocation,c.groups).submit();
      const checks={};for(const [name,expected]of Object.entries(c.expected)){
        const actual=await runtime.read(buffers[name],expected.constructor);checks[name]=compareArrays(actual,expected,c.tolerance);
        if(!checks[name].pass)throw new Error(`${name} differs from independent reference: ${JSON.stringify(checks[name])}`);
      }
      return {checks};
    }finally{for(const resource of Object.values(buffers))runtime.destroyBuffer(resource);}
  });
  await run('Two parameter versions in one batch retain their own snapshots',async()=>{
    const values=runtime.createBuffer(new Float32Array(2));
    try{
      const kernel=await runtime.kernel('__global__ void stamp(float* values,unsigned int slot,float value){if(threadIdx.x==0u)values[slot]=value;}',{workgroupSize:[4,1,1]});
      const inv=kernel.bind({values},{slot:0,value:11});const batch=runtime.batch();batch.dispatch(inv,[1]);inv.setScalars({slot:1,value:22});batch.dispatch(inv,[1]);batch.submit();
      const output=await runtime.read(values);if(output[0]!==11||output[1]!==22)throw new Error(`Uniform snapshot regression: ${output}`);
    }finally{runtime.destroyBuffer(values);}
  });
  await run('Reusable hierarchical reduction, including empty and singleton inputs',async()=>{
    for(const n of [0,1,4099]){
      const data=Float32Array.from({length:n},(_,i)=>(i%13-6)*0.125),expected=data.reduce((a,b)=>a+b,0);
      const input=runtime.createBuffer(data);let plan;
      try{
        plan=await prepareReduction(runtime,sources,{input,n});
        plan.encode(runtime.batch()).submit();
        const result=await runtime.read(plan.output);
        if(result[0]!==expected)throw new Error(`Reduction n=${n}: ${result[0]} != ${expected}`);
      }finally{plan?.dispose();runtime.destroyBuffer(input);}
    }
  });
  // Explicitly exercise workgroup variants used by the tuner, beyond the catalogue defaults.
  for(const block of [64,256])await run(`SAXPY workgroup specialization ${block}`,async()=>{
    const n=1031,xData=Float32Array.from({length:n},(_,i)=>i*0.125),x=runtime.createBuffer(xData),y=runtime.createBuffer(n*4);
    try{const kernel=await runtime.kernel(sources.saxpy,{entry:'saxpy',workgroupSize:[block,1,1]});runtime.batch().dispatch(kernel.bind({x,y},{a:2,n}),[Math.ceil(n/block)]).submit();const out=await runtime.read(y);for(let i=0;i<n;i++)if(out[i]!==xData[i]*2)throw new Error(`Mismatch at ${i}`);}finally{runtime.destroyBuffer(x);runtime.destroyBuffer(y);}
  });
  return {schema:'cuda-webshader.correctness.v1',mode:'REAL WebGPU execution',device:runtime.describe(),date:new Date().toISOString(),total:results.length,passed:results.filter(r=>r.pass).length,failed:results.filter(r=>!r.pass).length,wallMs:performance.now()-start,results};
}
