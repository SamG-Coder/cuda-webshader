import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';

test('Null-initialized pointer is rebound for each channel and batch',()=>{
  const source='__global__ void k(float*out){int x=threadIdx.x;float*p=NULL;for(int i=0;i<2;i++){for(int channel=0;channel<3;channel++){p=out+i*12+channel*4+x;*p=float(i*100+channel*10+x);}}}';
  const artifact=compile(source,{workgroupSize:[4,1,1]}),out=new Float32Array(24);
  executeCPU(artifact,{out},{},[1]);
  assert.deepEqual([...out],Array.from({length:24},(_,i)=>Math.floor(i/12)*100+Math.floor(i%12/4)*10+i%4));
  assert.equal(artifact.metadata.bindings[0].readOnly,false);
});
test('Both branches may initialize a deferred pointer',()=>{
  const c=compile('__global__ void k(int*out){int*p=0;if(threadIdx.x==0)p=out;else p=out+1;*p=7;}',{workgroupSize:[2,1,1]}),out=new Int32Array(2);
  executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[7,7]);
});
test('Deferred pointers reject reads without definite initialization and binding changes',()=>{
  for(const body of [
    'int*p=NULL;*p=1;p=out;',
    'int*p=NULL;if(threadIdx.x==0)p=out;*p=1;',
    'int*p=NULL;for(int i=0;i<0;i++)p=out;*p=1;',
    'int*p=NULL;p=p+1;',
    'int*p=NULL;p=out;p=other;*p=1;',
    'int*p=NULL;p=out;p=NULL;',
    'int*p=NULL;{int*p=out;}p=out;',
    'int*p=NULL;{int*out=other;p=out;}',
    'int NULL=1;int*p=NULL;p=out;*p=1;',
  ])assert.throws(()=>compile('__global__ void k(int*out,int*other){'+body+'}'),/pointer|Pointer/);
});
