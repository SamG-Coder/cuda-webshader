import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Double local quadratic solve matches native CUDA',()=>{
 const a=compile(readFileSync(new URL('double-locals.cu',import.meta.url),'utf8'),{entry:'doubleLocals',workgroupSize:[128]}),n=129,input=Float32Array.from({length:n},(_,i)=>(i+1)*.125),output=new Float32Array(n*3),pairs=Float32Array.from({length:n*2},(_,i)=>i%2?-Math.floor(i/2):i/2);input[n-1]=1e20;
 executeCPU(a,{input,output},{n},[2]);
 const reference=readFileSync(new URL('../reports/double-locals-native.bin',import.meta.url));
 assert.deepEqual(Buffer.from(output.buffer),reference.subarray(0,output.byteLength));
});
