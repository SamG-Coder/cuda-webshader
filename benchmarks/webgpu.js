import {GpuRuntime} from '../src/runtime/runtime.js';
import {loadKernelSources} from '../src/kernels.js';
import {prepareReduction} from '../src/runtime/operations.js';
import {compareArrays} from '../tests/cases.js';
import {makeComparisonCases} from './cases.js';
const median=a=>[...a].sort((a,b)=>a-b)[Math.floor(a.length/2)];
export async function runComparison(onResult=()=>{}) {
 const runtime=await GpuRuntime.create(),sources=await loadKernelSources(),device=runtime.device,results=[];
 if(!device.features.has('timestamp-query'))throw new Error('Cross-API GPU comparison requires timestamp-query');
 const querySet=device.createQuerySet({type:'timestamp',count:2}),resolve=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),readback=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{for(const scale of ['small','large'])for(const c of makeComparisonCases(scale)){
  const buffers=Object.fromEntries(Object.entries(c.buffers).map(([k,v])=>[k,runtime.createBuffer(v)]));let plan;
  try{
   const kernel=await runtime.kernel(sources[c.id],{entry:c.id,workgroupSize:c.block});let encode;
   if(c.id==='reduce_sum'){plan=await prepareReduction(runtime,sources,{input:buffers.input,n:c.scalars.n});buffers.output=plan.output;encode=b=>plan.encode(b);}
   else {const invocation=kernel.bind(buffers,c.scalars);encode=b=>b.dispatch(invocation,c.groups);
    if(c.id==='histogram'){const clear=await runtime.kernel('__global__ void clear_bins(unsigned int* bins) { bins[threadIdx.x]=0u; }',{entry:'clear_bins',workgroupSize:[256,1,1]});const ci=clear.bind({bins:buffers.bins},{});encode=b=>b.dispatch(ci,[1]).dispatch(invocation,c.groups);}
   }
   encode(runtime.batch()).submit();const checks={};
   for(const [k,expected] of Object.entries(c.expected)){checks[k]=compareArrays(await runtime.read(buffers[k],expected.constructor),expected,{absolute:expected instanceof Uint32Array?0:0.001,relative:expected instanceof Uint32Array?0:0.001});if(!checks[k].pass)throw new Error(`${c.key}: ${JSON.stringify(checks)}`);}
   const reset=()=>{for(const [k,v] of Object.entries(c.buffers))runtime.write(buffers[k],v);};
   reset();let b=runtime.batch();for(let i=0;i<c.warmup;i++)encode(b);b.submit();await runtime.idle();const timings=[];
   for(let s=0;s<c.samples;s++){reset();await runtime.idle();const start=performance.now();b=runtime.batch({timestampWrites:{querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}});for(let i=0;i<c.iterations;i++)encode(b);b.endPass();b.encoder.resolveQuerySet(querySet,0,2,resolve,0);b.encoder.copyBufferToBuffer(resolve,0,readback,0,16);b.submit();await runtime.idle();const wallMs=(performance.now()-start)/c.iterations;await readback.mapAsync(GPUMapMode.READ);const ts=new BigUint64Array(readback.getMappedRange()),gpuMs=Number(ts[1]-ts[0])/1e6/c.iterations;readback.unmap();if(gpuMs<=0)throw new Error('Zero timestamp: increase iterations');timings.push({gpuMs,wallMs});}
   const r={key:c.key,id:c.id,scalars:c.scalars,block:c.block,groups:c.groups,iterations:c.iterations,samples:c.samples,warmup:c.warmup,checks,gpuMedianMs:median(timings.map(t=>t.gpuMs)),wallMedianMs:median(timings.map(t=>t.wallMs)),timings};results.push(r);onResult(r);
  }finally{if(plan){delete buffers.output;plan.dispose();}for(const b of Object.values(buffers))runtime.destroyBuffer(b);}
 }
 return {date:new Date().toISOString(),device:runtime.describe(),userAgent:navigator.userAgent,results};
 }finally{querySet.destroy();resolve.destroy();readback.destroy();runtime.dispose();}
}
