import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=readFileSync(new URL('device-heap.cu',import.meta.url),'utf8');
test('Device allocation requires bounded persistent pools and cannot run in the CPU oracle',()=>{
 const options={entry:'allocate_values',objectHeap:'persistent',deviceHeap:{maxAllocations:256,maxElements:32}};
 const a=compile(source,options);assert.equal(a.metadata.objectHeap.types[0].byteLength,66560);
 assert.throws(()=>executeCPU(a,{},{}),/Persistent object arenas/);
 for(const deviceHeap of [true,{}, {maxAllocations:0,maxElements:32},{maxAllocations:256,maxElements:1.5},{maxAllocations:65536,maxElements:65536}])assert.throws(()=>compile(source,{...options,deviceHeap}));
 assert.throws(()=>compile(source,{...options,objectHeap:'invocation'}),/persistent/);
});
test('Allocation output casts cannot escape into arbitrary expressions',()=>{
 assert.throws(()=>compile('struct A {float2* p;}; __global__ void k(A* a){float x=(void**)&a[0].p;}'),/Unsupported|Unknown/);
});
test('Nested pointer records participate in arena ownership',()=>{
 const a=compile('struct A {float2* p;};struct B {A nested;};__global__ void k(B* records){cudaMalloc((void**)&records[0].nested.p,8u);}',{objectHeap:'persistent',deviceHeap:{maxAllocations:2,maxElements:2}});
 assert.deepEqual(a.metadata.objectHeap.pointerBuffers,['records']);
});
test('Child-launch syntax is retained while parent execution is explicitly rejected',()=>{
 const s='__global__ void child(float* x){x[threadIdx.x]=1.f;} __global__ void parent(float* x){child<<<1,32>>>(x);}';
 const a=compile(s,{entry:'child',workgroupSize:[32]});
 // The parser retains the child launch separately from ordinary helper calls.
 assert.ok(JSON.stringify(a.ast.functions.find(f=>f.name==='parent').body).includes('device-launch'));
 assert.throws(()=>compile(s,{entry:'parent'}),/GPU scheduling/);
});
