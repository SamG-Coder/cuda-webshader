import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
test('Original Bezier record retains a typed pointer field with explicit WebGPU layout',()=>{
 const a=compile(readFileSync(new URL('bezier-pointer-record.cu',import.meta.url),'utf8'),{workgroupSize:[64]});
 assert.equal(a.metadata.bindings.find(b=>b.name==='records').stride,32);
 const f=a.ast.structs.find(s=>s.name==='BezierLine').fields.find(f=>f.name==='vertexPos');
 assert.equal(f.pointerElement,'vec2<f32>');
});
test('Pointer fields reject dereference until device heap support exists',()=>{
 assert.throws(()=>compile('struct R {float2* p;}; __global__ void k(R* r){r[0].p[0]=make_float2(1.f,2.f);}'),/device heap allocation/);
});
test('Pointer identities cannot be fabricated or mixed between element types',()=>{
 for(const body of ['r[0].p=1u;','r[0].p=r[0].q;','r[0].p+=1u;'])assert.throws(()=>compile('struct R {float2* p;float* q;}; __global__ void k(R* r){'+body+'}'));
});
