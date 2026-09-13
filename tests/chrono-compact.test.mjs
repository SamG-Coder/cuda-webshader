import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=readFileSync(new URL('chrono-compact.cu',import.meta.url),'utf8');
test('Original active-list kernel consumes positive prefixes and leaves unused slots untouched',()=>{
 const activity=new Int32Array([-1,1,0,1,-1,0,1]),positive=new Uint32Array(7),prefixSum=new Uint32Array([0,0,1,1,2,2,2]),activeListD=new Uint32Array(7).fill(0xffffffff);
 executeCPU(compile(source,{entry:'normalizeActivity',workgroupSize:[4]}),{activity,positive},{n:7},[2]);
 assert.deepEqual([...positive],[0,1,0,1,0,0,1]);
 executeCPU(compile(source,{entry:'fillActiveListD',workgroupSize:[4]}),{prefixSum,extendedActivityIdD:activity,activeListD},{numAllMarkers:7},[2]);
 assert.deepEqual([...activeListD],[1,3,6,0xffffffff,0xffffffff,0xffffffff,0xffffffff]);
 const positions=Float32Array.from({length:28},(_,i)=>i+.25),selected=new Float32Array(12);
 executeCPU(compile(source,{entry:'gatherSelected',workgroupSize:[4]}),{positions,activeList:activeListD,selected},{n:3},[1]);
 assert.deepEqual([...selected],[...positions.slice(4,8),...positions.slice(12,16),...positions.slice(24,28)]);
});
