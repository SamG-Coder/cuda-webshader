import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
const source=readFileSync(new URL('./float4-surface-kernel.cuh',import.meta.url),'utf8');
test('float4 1D surface uses 16-byte offsets and one-row rgba32float storage',()=>{
 const artifact=compile(source,{entry:'writeTransfer',workgroupSize:[32,1,1]});
 assert.equal(artifact.metadata.surfaces[0].format,'rgba32float');
 assert.equal(artifact.metadata.surfaces[0].coordinates,'global-x');
 assert.match(artifact.wgsl,/texture_storage_2d<rgba32float, write>/);
 assert.doesNotThrow(()=>compile(source,{entry:'readTransferZero',workgroupSize:[1,1,1]}));
 assert.doesNotThrow(()=>compile(source.replace('x * sizeof(float4)','x * 16').replace('x * 16);','x * 16, cudaBoundaryModeTrap);'),{entry:'writeTransfer'}));
});
test('float4 surface rejects unsafe byte offsets, changing coordinates and boundary modes',()=>{
 for(const offset of ['x','x * 4','(x + 1) * 16']) assert.throws(()=>compile(source.replace('x * sizeof(float4)',offset),{entry:'writeTransfer'}),/offsets cannot be checked/);
 assert.throws(()=>compile(source.replace('if (x < count) surf1Dwrite','x++; if (x < count) surf1Dwrite'),{entry:'writeTransfer'}),/offsets cannot be checked/);
 assert.throws(()=>compile(source.replace('x * sizeof(float4));','x * sizeof(float4), cudaBoundaryModeClamp);'),{entry:'writeTransfer'}),/trap mode/);
});
test('original volume integration compiles unchanged with emulated double expressions',()=>{
 const original=readFileSync(new URL('./volume-transfer-kernel.cuh',import.meta.url),'utf8');
 const artifact=compile(original,{entry:'d_integrate_trapezoidal'});assert.match(artifact.wgsl,/cw_d_div/);assert.match(artifact.wgsl,/cw_d_lt/);
});
