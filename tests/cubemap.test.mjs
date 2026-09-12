import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';
test('Cubemap lookup inference follows texture helpers and declares scalar face layers',()=>{
 const a=compile('__device__ float read(cudaTextureObject_t t,float x){return texCubemap<float>(t,x,0.2f,0.3f);}__global__ void k(float* out,cudaTextureObject_t tex){out[0]=read(tex,1.0f);}');
 assert.equal(a.metadata.textures[0].dimension,'2d-array');assert.equal(a.metadata.textures[0].format,'r32float');assert.equal(a.metadata.textures[0].coordinates,'cube-direction');assert.match(a.wgsl,/texture_2d_array<f32>/);assert.match(a.wgsl,/textureSampleLevel/);
});
test('Cubemaps reject incorrect value types, argument counts and mixed dimensions',()=>{
 for(const lookup of ['texCubemap<float4>(tex,1,0,0)','texCubemap<float>(tex,1,0)','texCubemap<float>(tex,make_float2(1,2),0,0)'])assert.throws(()=>compile('__global__ void k(float* out,cudaTextureObject_t tex){out[0]='+lookup+';}'),/Cubemap|cubemap/);
 assert.throws(()=>compile('__global__ void k(float* out,cudaTextureObject_t tex){out[0]=texCubemap<float>(tex,1,0,0)+tex3D<float>(tex,1,0,0);}'),/mix/);
});
