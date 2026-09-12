import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {packScalars} from '../src/runtime/runtime.js';
const source=readFileSync(new URL('./volume-preintegrate-kernel.cuh',import.meta.url),'utf8');
test('Original layered preintegration retains checked XY and uniform layer metadata',()=>{
 const c=compile(source,{workgroupSize:[8,8,1]});assert.deepEqual(c.metadata.surfaces[0].layer,{scalar:'layer'});assert.equal(c.metadata.surfaces[0].dimension,'2d-array');assert.equal(c.metadata.surfaces[0].format,'rgba32float');assert.match(c.wgsl,/texture_storage_2d_array/);
 for(const bad of [source.replace('x * sizeof(float4)','(x+1) * sizeof(float4)'),source.replace('y, layer);','y, layer + 1);'),source.replace('y, layer);','y, layer, cudaBoundaryModeClamp);')])assert.throws(()=>compile(bad));
});
test('Layered float4 sample format propagates through helpers and rejects mixed dimensionality',()=>{
 const s='__device__ float4 f(cudaTextureObject_t t,int l){return tex2DLayered<float4>(t,0,1,l);}__global__ void k(float4* out,cudaTextureObject_t t,int layer){out[0]=f(t,layer);}';
 const c=compile(s);assert.equal(c.metadata.textures[0].dimension,'2d-array');assert.match(c.wgsl,/texture_2d_array<f32>/);
 assert.throws(()=>compile(s.replace('out[0]=f(t,layer);','out[0]=f(t,layer)+tex1D<float4>(t,0);')),/cannot mix/);
 assert.throws(()=>compile(s.replace('0,1,l','0,1,0.5f')),/scalar coordinates/);
 assert.throws(()=>compile(s.replace('tex2DLayered<float4>','tex2DLayered<float>')),/requires a float4/);
});
test('Kernel scalar defaults are represented in the launch ABI and can be overridden',()=>{
 const c=compile('__global__ void k(float* out,float value=0.25f,int n=3,bool enabled=true){out[0]=enabled?value*n:0.f;}',{workgroupSize:[1]});
 assert.deepEqual(c.metadata.scalars.map(s=>s.defaultValue),[.25,3,1]);const defaults=new DataView(packScalars(c.metadata,{}));assert.equal(defaults.getFloat32(0,true),.25);assert.equal(defaults.getInt32(4,true),3);assert.equal(defaults.getUint32(8,true),1);
 const out=new Float32Array(1);executeCPU(c,{out},{},[1]);assert.equal(out[0],.75);executeCPU(c,{out},{value:.5},[1]);assert.equal(out[0],1.5);assert.throws(()=>packScalars(c.metadata,{value:undefined}),/finite/);
});
test('CUDA exp float overload is preserved without accepting double exp',()=>{
 const c=compile('__global__ void k(float* out){out[0]=exp(-1.f);}',{workgroupSize:[1]}),out=new Float32Array(1);executeCPU(c,{out},{},[1]);assert.equal(out[0],Math.fround(Math.exp(-1)));assert.throws(()=>compile('__global__ void k(float* out){out[0]=exp(-1.0);}'),/float overload/);
});
