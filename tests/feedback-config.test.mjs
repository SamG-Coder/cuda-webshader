import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {suggestConfig,validateConfig} from '../src/sandbox/config.js';
test('Sandbox feedback validates matching buffer shapes and rejects chains or cycles',()=>{
 const {metadata}=compile('__global__ void k(const float4* oldPos,float4* newPos,float* scalar){newPos[threadIdx.x]=oldPos[threadIdx.x];scalar[threadIdx.x]=1.0f;}'),base=suggestConfig({name:'k',metadata});base.feedback={oldPos:'newPos'};assert.doesNotThrow(()=>validateConfig(base,metadata));
 for(const feedback of [{oldPos:'oldPos'},{missing:'newPos'},{oldPos:'scalar'},{oldPos:'newPos',newPos:'oldPos'},[]])assert.throws(()=>validateConfig({...base,feedback},metadata),/Feedback/);
 const mismatch=structuredClone(base);mismatch.buffers.oldPos.records--;assert.throws(()=>validateConfig(mismatch,metadata),/matching/);
});
