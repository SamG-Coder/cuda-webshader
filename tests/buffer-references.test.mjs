import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';
const source=readFileSync('tests/captured-points.cu','utf8'),options={entry:'bind_points',valueBuffers:['pts'],objectHeap:'persistent',workgroupSize:[1]};
test('Captured scalar buffer references preserve stable arena imports and record ownership',()=>{
 const a=compile(source,options),b=compile(source,{...options,entry:'read_points'});assert.deepEqual(a.metadata.objectHeap,b.metadata.objectHeap);assert.deepEqual(a.metadata.objectHeap.imports.map(i=>i.name),['x0','y0','x1','y1']);assert.deepEqual(a.metadata.objectHeap.pointerBuffers,['pts']);assert.equal(a.metadata.bindings[0].stride,8);
 assert.throws(()=>compile(source,{...options,objectHeap:'invocation'}),/persistent/);
 assert.throws(()=>compile(source.replace('float* x0','int* x0'),options),/Overload|matching/);
});
test('Captured pointer null tests are valid but arithmetic and invented handles are rejected',()=>{
 const s='class View {float* data;public:__device__ View(float* p):data(p){}__device__ bool valid()const{return data!=NULL;}};__global__ void k(float* input,int* out){View v(input);out[0]=v.valid();}';
 assert.ok(compile(s,{entry:'k',objectHeap:'persistent'}).wgsl);
 assert.throws(()=>compile(s.replace('data!=NULL','data+1'),{entry:'k',objectHeap:'persistent'}),/pointer|Pointer/);
 assert.throws(()=>compile(s.replace('View v(input)','View v(123)'),{entry:'k',objectHeap:'persistent'}),/convert|Convert|Overload/);
});
