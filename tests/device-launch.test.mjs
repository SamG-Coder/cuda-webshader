import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';
const source='__global__ void child(float* out,int i){out[i]=1.f;} __global__ void parent(float* out){child<<<1,32>>>(out,2);}';
const options={entry:'parent',objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:4}};
test('GPU queue metadata retains the original child binding and scalar argument',()=>{
 const a=compile(source,options),q=a.metadata.deviceLaunchQueue.queues[0];assert.equal(a.metadata.deviceLaunchQueue.producerOnly,true);assert.equal(q.byteLength,80);assert.deepEqual(q.block,[32,1,1]);assert.deepEqual(q.buffers,[{name:'out',parent:'out',type:'f32'}]);assert.equal(q.scalars[0].name,'i');
});
test('Device queue bounds and launch interfaces are explicit',()=>{
 for(const maxLaunches of [0,-1,65536,1.5])assert.throws(()=>compile(source,{...options,deviceLaunchQueue:{maxLaunches}}),/maxLaunches/);
 assert.throws(()=>compile(source,{...options,objectHeap:'invocation'}),/persistent/);
 assert.throws(()=>compile(source.replace('>>>(out,2)','>>>(out+1,2)'),options),/named parent buffers/);
 assert.throws(()=>compile(source.replace('<<<1,32>>>','<<<1,32,16>>>'),options),/grid and block/);
 assert.throws(()=>compile(source.replace('<<<1,32>>>','<<<1,threadIdx.x>>>'),options),/constant/);
});
