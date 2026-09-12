import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=readFileSync(new URL('volatile-compound.cu',import.meta.url),'utf8');
test('Volatile shared updates use separate loads and stores for integers and floats',()=>{
 const a=compile(source,{workgroupSize:[32]}),out=new Int32Array(64),fout=new Float32Array(32);executeCPU(a,{out,fout},{},[1]);
 for(let i=0;i<32;i++){const lane=(i+1)%32;for(let row=0;row<2;row++)assert.equal(out[i*2+row],((row*100+lane+7)*3>>1)^5);assert.equal(fout[i],(lane+1)/2);}
 assert.match(a.wgsl,/atomicLoad\(cw_volatile_address_/);assert.doesNotMatch(a.wgsl,/atomicAdd|atomicExchange/);
});
test('Ordinary atomic storage still requires explicit atomic operations',()=>{assert.throws(()=>compile('__global__ void k(){__shared__ int s[32];atomicAdd(&s[threadIdx.x],1);s[threadIdx.x]+=2;}'),/explicit atomicAdd/);});
test('Volatile compound destinations retain the side-effect restriction',()=>{assert.throws(()=>compile(source.replace('rows[row][lane]+=7;','rows[row][lane++]+=7;'),{workgroupSize:[32]}),/assignment destinations/);});
