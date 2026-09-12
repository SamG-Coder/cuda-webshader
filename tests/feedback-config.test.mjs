import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {suggestConfig,validateConfig} from '../src/sandbox/config.js';
test('Sandbox feedback validates matching buffer shapes and rejects chains or cycles',()=>{
 const {metadata}=compile('__global__ void k(const float4* oldPos,float4* newPos,float* scalar){newPos[threadIdx.x]=oldPos[threadIdx.x];scalar[threadIdx.x]=1.0f;}'),base=suggestConfig({name:'k',metadata});base.feedback={oldPos:'newPos'};assert.doesNotThrow(()=>validateConfig(base,metadata));
 for(const feedback of [{oldPos:'oldPos'},{missing:'newPos'},{oldPos:'scalar'},{oldPos:'newPos',newPos:'oldPos'},[]])assert.throws(()=>validateConfig({...base,feedback},metadata),/Feedback/);
 const mismatch=structuredClone(base);mismatch.buffers.oldPos.records--;assert.throws(()=>validateConfig(mismatch,metadata),/matching/);
});
test('Scalar volume previews require a complete matching halo layout',()=>{
 const {metadata}=compile('__global__ void k(float* out){out[threadIdx.x]=1.0f;}'),config=suggestConfig({name:'k',metadata});config.buffers.out.records=120;config.volume={dimensions:[2,3,4],halo:1};assert.doesNotThrow(()=>validateConfig(config,metadata));
 for(const volume of [{dimensions:[2,3,4],halo:0},{dimensions:[2,0,4],halo:1},{dimensions:[2,3,4],halo:-1},{dimensions:[2,3],halo:1}])assert.throws(()=>validateConfig({...config,volume},metadata),/Volume/);
});
