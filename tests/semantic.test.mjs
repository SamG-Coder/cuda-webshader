import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {KERNELS} from '../src/kernels.js';import {makeCases,compareArrays,cloneBuffers} from './cases.js';
const compiled=Object.fromEntries(KERNELS.map(k=>[k.id,compile(readFileSync(new URL(`../kernels/${k.file}`,import.meta.url),'utf8'),{entry:k.id,workgroupSize:k.workgroupSize})]));
for(const c of makeCases())test(`CPU typed-AST reference: ${c.name}`,()=>{
  const buffers=cloneBuffers(c.buffers);executeCPU(compiled[c.id],buffers,c.scalars,c.groups);
  for(const [name,expected]of Object.entries(c.expected)){const result=compareArrays(buffers[name],expected,c.tolerance);assert.equal(result.pass,true,`${name}: ${JSON.stringify(result)}`);}
});
test('Hierarchical reduction covers several dispatch levels with exact binary-fraction inputs',()=>{
  let input=Float32Array.from({length:4099},(_,i)=>(i%13-6)*0.125),n=input.length;
  const expected=Array.from(input).reduce((a,b)=>a+b,0);
  while(n>1){const groups=Math.ceil(n/256),output=new Float32Array(groups);executeCPU(compiled.reduce_sum,{input,output},{n},[groups]);input=output;n=groups;}
  assert.equal(input[0],expected);
});
test('Oracle detects nonuniform barriers (GPU validator remains authoritative)',()=>{
 const c=compile('__global__ void bad(unsigned int n) { if(threadIdx.x<n) __syncthreads(); }',{workgroupSize:[4,1,1]});
 assert.throws(()=>executeCPU(c,{},{n:2},[1]),/Nonuniform/);
});
