import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {packScalars} from '../src/runtime/runtime.js';
test('Pitched float2 local aliases and direct dereferences retain byte and element offsets',()=>{
 const source='__global__ void k(float2*v,float2*out,size_t pitch){int row=1;float2*p=(float2*)((char*)v+row*pitch)+2;*p=make_float2(3.0f,4.0f);out[0]=*((float2*)((char*)v+row*pitch)+2);}';
 const a=compile(source,{workgroupSize:[1,1,1]}),v=new Float32Array(32).fill(-77),out=new Float32Array(2);executeCPU(a,{v,out},{pitch:64},[1]);assert.deepEqual([...out],[3,4]);assert.deepEqual([...v.slice(20,22)],[3,4]);assert.ok(v.every((x,i)=>i===20?x===3:i===21?x===4:x===-77));
 assert.throws(()=>packScalars(a.metadata,{pitch:63}),/multiple/);
});
test('Typed byte casts reject misalignment, reinterpretation and discarded constness',()=>{
 for(const source of [
 '__global__ void k(float*v){float*p=(float*)((char*)v+3);}',
 '__global__ void k(float*v){float2*p=(float2*)((char*)v+8);}',
 '__global__ void k(const float*v,float*out){out[0]=*((float*)((char*)v+4));}',
 '__global__ void k(float*v){*((const float*)((char*)v+4))=1.0f;}'
 ])assert.throws(()=>compile(source));
});
