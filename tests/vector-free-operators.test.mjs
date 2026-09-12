import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Original NVIDIA vector operators and a distinct overload body are preserved',()=>{
 const source=readFileSync(new URL('vector-free-operators.cu',import.meta.url),'utf8');
 const artifact=compile(source,{workgroupSize:[1]}),out=new Float32Array(8);
 executeCPU(artifact,{out},{},[1]);assert.deepEqual([...out],[1.75,2.5,-3,-4,4,7,2,5]);
});
test('Scalar-only operator declarations remain invalid',()=>{
 assert.throws(()=>compile('__device__ float operator+(float a,float b){return a-b;} __global__ void k(){}'),/class or vector/);
});
