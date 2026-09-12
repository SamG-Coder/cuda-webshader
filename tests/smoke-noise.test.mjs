import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
const helper=readFileSync(new URL('./smoke-noise-kernel.cuh',import.meta.url),'utf8'),probe=readFileSync(new URL('./smoke-noise-probe.cuh',import.meta.url),'utf8');
test('Original smoke noise helper propagates float4 texture format and 3D coordinate uniforms',()=>{
 const artifact=compile(helper+probe,{entry:'sampleSmokeNoise',workgroupSize:[128,1,1]});
 assert.equal(artifact.metadata.textures[0].format,'rgba32float');assert.equal(artifact.metadata.textures[0].dimension,'3d');assert.equal(artifact.metadata.textureScales[0].dimension,'3d');
 assert.match(artifact.wgsl,/fn cw_sample_float4_3d.*-> vec4<f32>/);assert.match(artifact.wgsl,/cw_scale_noiseTex: vec3<f32>/);
});
test('3D scalar and float4 texture formats cannot mix across a helper chain',()=>{
 assert.throws(()=>compile(helper+probe.replace('tex3D<float4>(texture,p.x,p.y,p.z).w','tex3D<float>(texture,p.x,p.y,p.z)')),/cannot mix/);
 assert.throws(()=>compile(helper.replace('tex3D<float4>','tex3D<float2>')+probe),/tex3D supports|cannot mix/);
 assert.throws(()=>compile(helper.replace('p.x, p.y, p.z','p.x, p.y, 1')+probe),/coordinates must be floats/);
});
