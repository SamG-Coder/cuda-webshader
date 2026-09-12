import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Buffer parameter shifts preserve aliases and valid dereferences from a negative halo origin',()=>{
 const c=compile('__global__ void k(const float* input,float* output){const float* saved=input;input-=2;input+=1;output[0]=input[1]+saved[2];}',{workgroupSize:[1]}),input=Float32Array.from([2,4,8]),output=new Float32Array(1);executeCPU(c,{input,output},{},[1]);assert.equal(output[0],10);assert.equal(c.metadata.bindings[0].readOnly,true);
});
test('Helper pointer shifts affect only the callee and accept mutable local aliases',()=>{
 const c=compile('__device__ float read(const float* p){p+=2;return p[0];}__global__ void k(const float* input,float* output){const float* p=input;p+=1;output[0]=read(p)+p[0];p-=1;output[1]=p[0];}',{workgroupSize:[1]}),input=Float32Array.from([1,2,4,8]),output=new Float32Array(2);executeCPU(c,{input,output},{},[1]);assert.deepEqual([...output],[10,1]);
});
test('Pointer shifts retain const pointee rules and reject invalid offset types or dereferences',()=>{
 assert.throws(()=>compile('__global__ void k(const float* p){p+=1;p[0]=2.0f;}'),/const/);
 assert.throws(()=>compile('__global__ void k(float* p){p+=1.5f;}'),/integer offset/);
 const c=compile('__global__ void k(float* p){p-=1;p[0]=1.0f;}',{workgroupSize:[1]});assert.throws(()=>executeCPU(c,{p:new Float32Array(4)},{},[1]),/bounds/);
});
test('Negative halo origins can be aliased and passed to helpers before valid dereferences',()=>{
 const c=compile('__device__ float read(const float* p){p+=2;return p[0];}__global__ void k(const float* input,float* out){input-=2;const float* alias=input;out[0]=read(alias);}',{workgroupSize:[1]}),out=new Float32Array(1);executeCPU(c,{input:Float32Array.from([7,8,9]),out},{},[1]);assert.equal(out[0],7);
});
