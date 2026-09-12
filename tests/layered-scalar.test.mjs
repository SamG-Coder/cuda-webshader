import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';
test('Scalar layered texture inference preserves float4 support and helper propagation',()=>{
 for(const [type,format] of [['float','r32float'],['float4','rgba32float']]){
 const a=compile('__device__ '+type+' read(cudaTextureObject_t t,int layer){return tex2DLayered<'+type+'>(t,0.5f,0.5f,layer);}__global__ void k('+type+'* out,cudaTextureObject_t tex){out[0]=read(tex,2);}');
 assert.equal(a.metadata.textures[0].dimension,'2d-array');assert.equal(a.metadata.textures[0].format,format);
 }
});
test('Layered textures reject mixed formats and noninteger layer selectors',()=>{
 assert.throws(()=>compile('__global__ void k(float* out,cudaTextureObject_t tex){out[0]=tex2DLayered<float>(tex,0,0,0)+tex2DLayered<float4>(tex,0,0,0).x;}'),/mix/);
 assert.throws(()=>compile('__global__ void k(float* out,cudaTextureObject_t tex){out[0]=tex2DLayered<float>(tex,0,0,0.5f);}'),/coordinates/);
});
