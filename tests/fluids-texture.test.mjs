import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
test('Original fluid texture alias produces rg32float metadata',()=>{
 const a=compile(readFileSync(new URL('./fluids-advectVelocity_k.cuh',import.meta.url),'utf8'),{entry:'advectVelocity_k',workgroupSize:[16,4,1]});
 assert.equal(a.metadata.textures[0].format,'rg32float');assert.equal(a.metadata.textures[0].dimension,'2d');assert.match(a.wgsl,/cw_sample_rgba2d\(.*\)\.xy/);
});
const source='typedef float2 Velocity;__device__ Velocity sample(cudaTextureObject_t tex,float2 p){return tex2D<Velocity>(tex,p.x,p.y);}__global__ void k(cudaTextureObject_t field,float2*out){out[0]=sample(field,make_float2(.25f,.75f));}';
test('Float2 sampling propagates through texture helpers with coordinate scaling',()=>{
 const a=compile(source);assert.equal(a.metadata.textures[0].format,'rg32float');assert.match(a.wgsl,/cw_scale_tex: vec2<f32>/);assert.match(a.wgsl,/cw_point_tex: f32/);
});
test('Texture chains reject mixing float2 and other sample formats',()=>{
 for(const type of ['float','float4'])assert.throws(()=>compile(source.replace('out[0]=sample(field,make_float2(.25f,.75f));','out[0]=sample(field,make_float2(.25f,.75f));'+type+' other=tex2D<'+type+'>(field,0.0f,0.0f);')),/cannot mix/);
});
