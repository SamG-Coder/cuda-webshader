import test from 'node:test';import assert from 'node:assert/strict';
import {emitThreaded} from '../src/wasm/compile.mjs';
test('WASM emits typed independent and cooperative entries from CUDA',async()=>{
 const source='__global__ void sum(const float* A,float* B,int n){__shared__ float tile[8];int i=(int)threadIdx.x;tile[i]=A[i];__syncthreads();if(i<n)B[i]=tile[7-i];}';
 const a=await emitThreaded(source,[{entry:'sum',workgroupSize:[8,1,1]}]);
 assert.equal(a.kernels[0].cooperative,true);assert.equal(a.kernels[0].params[0].constant,true);assert.equal(a.kernels[0].params[2].type,'i32');
 assert.match(a.cpp,/co_await CwBarrier\{0\}/);assert.match(a.cpp,/float tile\[8\]/);assert.match(a.cpp,/extern "C" int cw_sum/);
});
test('WASM rejects unsupported workgroup primitives and invalid specifications',async()=>{
 await assert.rejects(()=>emitThreaded('',[]),/specification/);
 await assert.rejects(()=>emitThreaded('',[{entry:'a'},{entry:'a'}]),/Duplicate/);
 await assert.rejects(()=>emitThreaded('__global__ void a(unsigned int* p){atomicAdd(p,1u);}',[{entry:'a',workgroupSize:[1,1,1]}]),/Atomics/);
});
