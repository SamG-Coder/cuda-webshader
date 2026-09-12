import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {packScalars} from '../src/runtime/runtime.js';
const source=readFileSync(new URL('chrono-hash.cu',import.meta.url),'utf8'),params=JSON.parse(readFileSync(new URL('../reports/chrono-params.json',import.meta.url)));
test('Original Chrono grid helpers retain the complete constant parameter record',()=>{
 const artifact=compile(source,{entry:'hashProbe',workgroupSize:[128]});
 assert.equal(artifact.metadata.bindings.find(x=>x.name==='points').stride,12);
 assert.equal(artifact.metadata.scalars.length,158);
 for(const scalar of artifact.metadata.scalars)if(scalar.origin==='constant')assert.ok(Object.hasOwn(params,scalar.name),scalar.name);
 assert.doesNotThrow(()=>packScalars(artifact.metadata,{...params,n:16739}));
 assert.match(source,/double pressure_height;/);assert.match(artifact.wgsl,/cw_field_pressure_height: cw_f64/);
});
test('Double constant fields expose exact low/high words without enabling incompatible double buffers',()=>{
 const s='struct Params {double x;};__constant__ static Params p;__global__ void k(float* out){out[0]=(float)(p.x-1.0);}';
 const a=compile(s);assert.deepEqual(a.metadata.scalars.map(p=>p.name),['constant.p.x.lo','constant.p.x.hi']);
 assert.match(a.wgsl,/cw_d_sub/);assert.throws(()=>compile('struct P {double x;};__global__ void k(P* p){p[0].x=1.0;}'),/host-shareable/);
 assert.throws(()=>compile('__global__ void k(float* out){out[0]=(float)floor(1.0);}'),/float overload/);
});
