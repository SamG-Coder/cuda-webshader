import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
const source=readFileSync(new URL('chrono-neighbors.cu',import.meta.url),'utf8');
test('Chrono return phases preserve top-level barriers without disabling validation',()=>{
 const a=compile(source,{entry:'findCellStartEndD',workgroupSize:[128],sharedMemoryBytes:516,predicatedReturns:true});
 assert.equal(a.metadata.predicatedReturns,true);assert.match(a.wgsl,/workgroupBarrier/);assert.doesNotMatch(a.wgsl,/diagnostic\(off/);
});
test('Return phases reject nested barriers, nested exits and unsupported local lifetimes',()=>{
 for(const body of ['if(threadIdx.x) {__syncthreads();} __syncthreads();','if(threadIdx.x){out[0]=1;return;} __syncthreads();','float a[2]; __syncthreads();','for(int i=0;i<2;i++){out[i]=1;} __syncthreads();']){
  assert.throws(()=>compile('__global__ void k(float* out){'+body+'}',{predicatedReturns:true}),/Predicated|Unsupported statement/);
 }
 assert.throws(()=>compile('__global__ void k(){}',{predicatedReturns:'yes'}),/must be boolean/);
});
test('Return predicate names cannot collide with user locals',()=>{
 const a=compile('__global__ void k(int* out){int cw_return_active=7; if(threadIdx.x>0)return; __syncthreads();out[0]=cw_return_active;}',{predicatedReturns:true});
 assert.match(a.wgsl,/cw_return_active_/);
});
