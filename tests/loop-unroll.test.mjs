import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';

const kernel=(loop)=>`__global__ void sample(float* out,unsigned count){float sum=0.0f;\n${loop}\nout[0]=sum;}`;
test('CUDA unroll expands bounded loops while preserving iteration scope and order',()=>{
  const artifact=compile(kernel('#pragma unroll\nfor(unsigned j=0u;j<8u;j+=1u){float term=(float)j;sum=sum*2.0f+term;}'));
  assert.doesNotMatch(artifact.wgsl,/\bloop\s*\{/);
  assert.equal((artifact.wgsl.match(/var v_term/g)||[]).length,8);
  const out=new Float32Array(1);executeCPU(artifact,{out},{count:8},[1]);assert.equal(out[0],247);
});
test('Unroll hints retain dynamic, oversized, escaping and mutated-counter loops',()=>{
  for(const loop of [
    'for(unsigned j=0u;j<count;j++)sum+=(float)j;',
    'for(unsigned j=0u;j<1000u;j++)sum+=(float)j;',
    'for(unsigned j=0u;j<8u;j++){if(j==3u)break;sum+=(float)j;}',
    'for(unsigned j=0u;j<8u;j++){if(j==3u)continue;sum+=(float)j;}',
    'for(unsigned j=0u;j<8u;j++){j+=1u;sum+=(float)j;}',
    'for(unsigned j=0u;j<8u;j++){unsigned j=2u;sum+=(float)j;}',
  ])assert.match(compile(kernel('#pragma unroll\n'+loop)).wgsl,/\bloop\s*\{/);
});
test('Explicit full factors expand and factor one disables expansion',()=>{
  for(const [hint,expanded] of [['4',true],['1',false],['2',false]]){
    const wgsl=compile(kernel(`#pragma unroll ${hint}\nfor(int j=1;j<=4;j++)sum+=(float)j;`)).wgsl;
    assert.equal(/\bloop\s*\{/.test(wgsl),!expanded);
  }
  assert.throws(()=>compile(kernel('#pragma unroll\nsum=1.0f;')),/precede a loop/);
});

test('A reference helper can change the counter and must retain loop control',()=>{
  const code='__device__ void advance(unsigned& j){j+=1u;}\n'+kernel('#pragma unroll\nfor(unsigned j=0u;j<8u;j++){advance(j);sum+=(float)j;}');
  assert.match(compile(code).wgsl,/\bloop\s*\{/);
});
