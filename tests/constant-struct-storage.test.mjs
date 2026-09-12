import test from 'node:test';import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source='struct Row {float x;int y;unsigned int z;};__constant__ Row rows[2];__device__ float read(int i){return rows[i].x+(float)rows[i].y+(float)rows[i].z;}__global__ void k(float* out){out[threadIdx.x]=read(threadIdx.x);}';
test('Constant struct storage preserves mixed 32-bit field layout through helpers',()=>{
 const a=compile(source,{workgroupSize:[2]}),data=new Uint32Array(6),v=new DataView(data.buffer);v.setFloat32(0,1.5,true);v.setInt32(4,-3,true);v.setUint32(8,9,true);v.setFloat32(12,2.25,true);v.setInt32(16,10,true);v.setUint32(20,20,true);
 const b=a.metadata.bindings.find(b=>b.name==='rows');assert.equal(b.stride,12);assert.equal(b.minBindingSize,24);assert.equal(b.readOnly,true);assert.deepEqual(b.fields.map(f=>f.offset),[0,4,8]);
 const out=new Float32Array(2);executeCPU(a,{out,rows:data},{},[1]);assert.deepEqual([...out],[7.5,32.25]);assert.throws(()=>executeCPU(a,{out,rows:new Uint32Array(3)},{},[1]),/too small/);
});
test('Constant struct arrays reject writes and layouts without a supported CUDA ABI',()=>{
 assert.throws(()=>compile(source.replace('out[threadIdx.x]=read(threadIdx.x);','rows[0].x=2.0f;')),/const/);
 for(const fields of ['float2 x;','float x[2];'])assert.throws(()=>compile('struct S {'+fields+'};__constant__ S a[2];__global__ void k(){}'),/flat/);
 assert.throws(()=>compile('struct S {float x;};__constant__ S a[2]={};__global__ void k(){}'),/uninitialized/);
});
