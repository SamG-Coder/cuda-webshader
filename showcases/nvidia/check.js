import {GpuRuntime} from '../../src/runtime/runtime.js';
import {fixture} from './fixtures.js';
export async function check(row,runtime){
 const f=fixture(row),resources={};
 try{for(const [name,data] of Object.entries(f.buffers))resources[name]=runtime.createBuffer(data);
 const kernel=await runtime.kernel(row.artifact);runtime.batch().dispatch(kernel.bind(resources,f.scalars),f.groups).submit();
 let maxError=0,failures=0,values=0,output;const outputs={};
 for(const [name,expected] of Object.entries(f.expectedOutputs||{[f.out]:f.expected})){
 const actual=await runtime.read(resources[name],f.buffers[name].constructor);let errors=0,maximum=0;
 actual.forEach((v,i)=>{const err=Math.abs(v-expected[i]);maximum=Math.max(maximum,err);if(!Number.isFinite(v)||err>(f.absoluteTolerance??3e-6)+(f.relativeTolerance??0)*Math.abs(expected[i]))errors++;});
 outputs[name]={values:actual.length,maxError:maximum,failures:errors};values+=actual.length;failures+=errors;maxError=Math.max(maxError,maximum);if(name===f.out)output=Array.from(actual.slice(0,128));
 }
 return {pass:failures===0,values,maxError,failures,output,outputs};
 }finally{await runtime.idle();for(const r of Object.values(resources))runtime.destroyBuffer(r);}
}
export async function checkAll(){const runtime=await GpuRuntime.create(),results=[];try{const adapter=runtime.describe();if(!/nvidia/i.test(JSON.stringify(adapter)))throw Error('NVIDIA hardware adapter required for this audit');for(const row of await(await fetch('./artifacts.json')).json()){try{results.push({sample:row.sample,entry:row.entry,...await check(row,runtime)});}catch(e){results.push({sample:row.sample,entry:row.entry,pass:false,error:e.message});}}return {adapter,softwareAdapterRequested:false,date:new Date().toISOString(),results};}finally{runtime.dispose();}}
