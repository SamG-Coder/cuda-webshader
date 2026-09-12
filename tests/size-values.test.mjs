import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {packScalars} from '../src/runtime/runtime.js';
const source=readFileSync(new URL('./size-values-kernel.cuh',import.meta.url),'utf8');
test('size_t parameters retain wide expression arithmetic with bounded launch values',()=>{
 const a=compile(source,{workgroupSize:[1,1,1]}),words=new Uint32Array(5),values=new Float32Array(2);
 assert.deepEqual(a.metadata.scalars[0],{name:'pitch',type:'u32',sourceType:'size_t',offset:0});
 executeCPU(a,{words,values},{pitch:0xffffffff,row:2},[1]);assert.deepEqual([...words],[0xfffffffe,1,0xfffffffd,1,8]);assert.equal(values[0],Math.fround(8589934590));assert.equal(values[1],Math.fround(12884901885));
 for(const pitch of [-1,1.5,4294967296,NaN])assert.throws(()=>packScalars(a.metadata,{pitch,row:1}));
});
test('Unsupported size_t memory layouts and arithmetic fail explicitly',()=>{
 for(const s of ['__global__ void k(size_t* p){}','struct S{size_t x;};__global__ void k(unsigned*out){S s;}','__global__ void k(unsigned*out){size_t local[2];}','__global__ void k(unsigned*out){__shared__ size_t v;}','__global__ void k(unsigned*out,size_t pitch){out[0]=(unsigned)(pitch/3);}'])assert.throws(()=>compile(s));
});
