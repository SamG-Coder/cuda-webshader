import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=readFileSync(new URL('native-records.cu',import.meta.url),'utf8');
test('Native bool records preserve padding, adjacent flags, float3 layout and helper offsets',()=>{
 const a=compile(source,{workgroupSize:[1]});const binding=a.metadata.bindings[0];
 assert.equal(binding.stride,24);assert.deepEqual(binding.nativeLayout.fields.map(f=>f.offset),[0,1,4,16,20]);
 const data=new Uint8Array(48).fill(0xa5),v=new DataView(data.buffer);
 for(let i=0;i<2;i++){v.setUint8(i*24,i);v.setUint8(i*24+1,1-i);v.setUint8(i*24+16,i);for(let j=0;j<3;j++)v.setFloat32(i*24+4+j*4,1+i*3+j,true);v.setFloat32(i*24+20,i?11:7,true);}
 const output=new Float32Array(4),flags=new Int32Array([0,1,2,3,4]);executeCPU(a,{data,output,flags},{},[1]);
 assert.deepEqual([...output],[23,127,7,1]);assert.deepEqual([...flags],[0,4,6,8,1]);
});
test('Packed records reject mutable pointers, partial strides and unsupported field types',()=>{
 for(const s of ['struct A{bool a;float x;};__global__ void k(A* a){}','struct A{bool a;};__global__ void k(const A* a){}','struct A{bool a;double x;};__global__ void k(const A* a){}','struct A{bool a;float x;};__global__ void k(const A* a){a[0].x=1.f;}'])assert.throws(()=>compile(s));
 const a=compile(source,{workgroupSize:[1]});assert.throws(()=>executeCPU(a,{data:new Uint8Array(47),output:new Float32Array(4),flags:new Int32Array(5)},{},[1]),/incomplete records/);
 assert.throws(()=>compile('__device__ void f(int& a){a=1;}__global__ void k(const int* a){f(a[0]);}'),/mutable|const/);
});
test('Original Chrono activity kernel uses native domain stride and exact scalar launch types',()=>{
 const a=compile(readFileSync(new URL('chrono-activity.cu',import.meta.url),'utf8'),{entry:'UpdateActivityD',workgroupSize:[128]});
 assert.equal(a.metadata.bindings.find(b=>b.name==='ad_body_D').stride,52);
 assert.deepEqual(a.metadata.bindings.find(b=>b.name==='ad_body_D').nativeLayout.fields.map(f=>f.offset),[0,4,16,28,40]);
 for(const name of ['time.lo','time.hi','constant.countersD.numAllMarkers.lo','constant.countersD.numAllMarkers.hi'])assert.ok(a.metadata.scalars.some(s=>s.name===name));
});
