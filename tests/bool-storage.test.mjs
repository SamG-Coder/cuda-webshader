import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('CUDA bool byte layout and volatile flags match native including guard bytes',()=>{
 const artifact=compile(readFileSync(new URL('bool-storage.cu',import.meta.url),'utf8'),{entry:'boolStorage',workgroupSize:[128]}),n=1025;
 const input=Uint8Array.from({length:n},(_,i)=>Number(i%3===1)),output=new Uint8Array(1032).fill(165),flag=new Uint8Array(4).fill(165),observed=new Int32Array(n);
 executeCPU(artifact,{input,output,flag,observed},{n},[9]);
 assert.deepEqual(Buffer.concat([Buffer.from(output),Buffer.from(flag),Buffer.from(observed.buffer)]),readFileSync(new URL('../reports/bool-storage-native.bin',import.meta.url)));
 for(const name of ['input','output','flag'])assert.equal(artifact.metadata.bindings.find(b=>b.name===name).stride,1);
 assert.equal(artifact.metadata.bindings.find(b=>b.name==='flag').atomic,true);
});
