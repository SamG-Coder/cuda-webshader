import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {packScalars} from '../src/runtime/runtime.js';import {kernelSource} from '../src/sandbox/import.js';
test('Constant arrays retain dynamic indexing, default zero filling and scalar overrides',()=>{
 const c=compile('__constant__ float coefficients[4]={1.0f,-2.0f,};__device__ float sample(int i){return coefficients[i];}__global__ void k(float* out){out[threadIdx.x]=sample((int)threadIdx.x);}',{workgroupSize:[4]}),out=new Float32Array(4);
 executeCPU(c,{out},{'constant.coefficients[2]':3},[1]);assert.deepEqual([...out],[1,-2,3,0]);const packed=new DataView(packScalars(c.metadata,{'constant.coefficients[2]':3}));assert.equal(packed.getFloat32(8,true),3);
});
test('Integer constant arrays preserve 32-bit values and reject invalid updates without mutation',()=>{
 const c=compile('__constant__ unsigned int a[2]={4294967295u,2147483648u};__constant__ int b[2]={-2147483647,7};__global__ void k(unsigned int* out){out[threadIdx.x]=a[threadIdx.x]+(unsigned int)b[threadIdx.x];}',{workgroupSize:[2]}),out=new Uint32Array(2);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[2147483648,2147483655]);const data=packScalars(c.metadata,{}),before=data.slice(0);assert.throws(()=>packScalars(c.metadata,{'constant.a[0]':-1},data),/u32/);assert.deepEqual(data,before);
});
test('Desktop extraction retains initialized constant arrays and rejects writes and unsupported shapes',()=>{
 const source='#include <cuda_runtime.h>\n__constant__ float a[2]={1.0f,2.0f};__global__ void k(float* out){out[0]=a[1];}';assert.equal(compile(kernelSource(source).source).metadata.scalars.length,2);
 for(const source of ['__constant__ float a[2];__global__ void k(){a[0]=1.0f;}','__constant__ float a[2][2][2];__global__ void k(){}','__constant__ float a[257];__global__ void k(float* out){out[0]=a[0];}','__constant__ float a[1]={1.0f,2.0f};__global__ void k(float* out){out[0]=a[0];}'])assert.throws(()=>compile(source),/constant|Constant/);
});
