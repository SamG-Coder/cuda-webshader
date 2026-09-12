import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {kernelSource} from '../src/sandbox/import.js';
test('Device arrays persist across kernels and remain visible in helpers',()=>{
 const source='static __device__ float values[4];__device__ void inc(){values[threadIdx.x]+=2.0f;}__global__ void k(){inc();}';
 const a=compile(source,{workgroupSize:[4]}),values=new Float32Array(4);
 assert.equal(a.metadata.bindings[0].readOnly,false);assert.equal(a.metadata.bindings[0].minBindingSize,16);
 executeCPU(a,{values},{},[1]);executeCPU(a,{values},{},[1]);assert.deepEqual([...values],[4,4,4,4]);
 assert.throws(()=>executeCPU(a,{values:new Float32Array(1)},{},[1]),/too small/);
 const extracted=kernelSource('#include <cuda_runtime.h>\n'+source+'\nint main(){return 0;}');assert.match(extracted.source,/static __device__ float values/);assert.doesNotThrow(()=>compile(extracted.source,{workgroupSize:[4]}));
});
test('Local and helper argument shadowing do not write global storage',()=>{
 const a=compile('__device__ int value[4];__device__ int read(int value){return value+1;}__global__ void k(int* out){int value=7;out[0]=read(value);}',{workgroupSize:[1]});
 const out=new Int32Array(1),value=new Int32Array(4);executeCPU(a,{out,value},{},[1]);assert.equal(out[0],8);assert.equal(a.metadata.bindings.find(b=>b.name==='value').readOnly,true);
});
test('Device arrays reject unsupported declarations and ambiguous binding names',()=>{
 for(const s of ['__device__ float x[0];','__device__ float x[16777217];','__device__ float x[4]= {};','__device__ float x[4][2];','__device__ float x[4];__constant__ float x;','__device__ float x[4];__device__ float x[4];'])assert.throws(()=>compile(s+'__global__ void k(){}'));
 assert.throws(()=>compile('__device__ float x[4];__global__ void k(float* x){}'),/shadow/);
 for(const body of ['x+=1;','x=x+1;'])assert.throws(()=>compile('__device__ int x[4];__global__ void k(){'+body+'}'),/Pointer/);
});
