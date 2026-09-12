import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';
const source='class Range{int begin,end;public:__device__ int count()const{return end-begin;}};__global__ void k(Range*nodes,int*out){const Range &node=nodes[blockIdx.x];int count=node.count();if(count<1)return;__syncthreads();out[blockIdx.x*32+threadIdx.x]=count;}';
const options={valueBuffers:['nodes'],workgroupSize:[32]};
test('Workgroup-indexed pure record getters publish a uniform snapshot',()=>{const a=compile(source,options);assert.match(a.wgsl,/workgroupUniformLoad/);assert.equal(a.metadata.workgroupStorageBytes,16);assert.doesNotMatch(a.wgsl,/@diagnostic/);});
test('Per-lane record addresses are never broadcast as uniform',()=>{const a=compile(source.replace('nodes[blockIdx.x]','nodes[threadIdx.x]'),options);assert.doesNotMatch(a.wgsl,/workgroupUniformLoad/);});
test('Mutated buffer pointers are not assumed to stay workgroup uniform',()=>{const a=compile(source.replace('const Range &node','nodes+=threadIdx.x;const Range &node'),options);assert.doesNotMatch(a.wgsl,/workgroupUniformLoad/);});
test('Getters with effects or lane dependencies are not broadcast',()=>{
 for(const body of ['return end-begin+threadIdx.x;','out_of_line();return end-begin;']){const changed=source.replace('return end-begin;',body),preamble=body.includes('out_of_line')?'__device__ void out_of_line(){}':'';const a=compile(preamble+changed,options);assert.doesNotMatch(a.wgsl,/workgroupUniformLoad/);}
});
