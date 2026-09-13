import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Switch selection, fall-through, nested control and selector effects match native CUDA',()=>{
 const a=compile(readFileSync(new URL('switch.cu',import.meta.url),'utf8'),{entry:'switchCases',workgroupSize:[9]}),output=new Int32Array(36);
 executeCPU(a,{input:new Int32Array([-3,-2,-1,0,1,2,3,4,5]),output},{},[1]);
 assert.deepEqual([...output],JSON.parse(readFileSync(new URL('../reports/switch-native.json',import.meta.url))));
});
test('Invalid switch selectors, labels and case scopes fail explicitly',()=>{
 for(const body of ['switch(1.f){case 1:break;}','switch(1){case 1:break;case 1:break;}','switch(1){default:break;default:break;}','switch(1){case 1.f:break;}','switch(1){case threadIdx.x:break;}','switch(1){case 1: int a=0;break;}','switch(1){case 1:continue;}'])assert.throws(()=>compile('__global__ void k(){'+body+'}'));
});
