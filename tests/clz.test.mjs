import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('CUDA clz uses unsigned bits, returns 32 for zero, and evaluates its argument once',()=>{
 const a=compile('__global__ void clz(const unsigned* input,int* output){unsigned i=threadIdx.x;output[i]=__clz(input[i]);}',{workgroupSize:[64,1,1]});
 assert.match(a.wgsl,/countLeadingZeros/);
 const input=Uint32Array.from({length:64},(_,i)=>i<32?2**i:(2**(i-32)-1)>>>0),output=new Int32Array(64);
 executeCPU(a,{input,output},{},[1]);assert.deepEqual([...output],[...input].map(Math.clz32));
 const b=compile('__global__ void clz(int* output){int v=-1;output[0]=__clz(v++);output[1]=v;output[2]=__clz(v);}');
 const result=new Int32Array(3);executeCPU(b,{output:result},{},[1]);assert.deepEqual([...result],[0,0,32]);
});
test('CUDA clz rejects invalid arity and non-integer operands',()=>{
 for(const argument of ['', '1,2','1.0f'])assert.throws(()=>compile(`__global__ void clz(int* out){out[0]=__clz(${argument});}`),/32-bit integer/);
});
