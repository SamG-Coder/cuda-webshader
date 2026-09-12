import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {kernelSource} from '../src/sandbox/import.js';import {matrixFixture} from '../showcases/nvidia/matrixmul-fixtures.js';
const source='template <int N> __global__ void k(int* out){int sum=0;for(int i=0,j=1;i<N;i++,j+=2){if(i==1)continue;sum+=j;}out[0]=sum;}';
test('Explicit integer specialization and ordered loop steps survive continue',()=>{const c=compile(source,{entry:'k<4>',workgroupSize:[1,1,1]}),out=new Int32Array(1);executeCPU(c,{out},{},[1]);assert.equal(out[0],13);assert.deepEqual(c.metadata.templateArguments,{N:4});assert.throws(()=>compile(source),/template entry/);});
test('Template values specialize shared dimensions and remain immutable',()=>{
 const c=compile('template<int N> __global__ void k(float* out){__shared__ float a[N][N];a[0][0]=1.0f;__syncthreads();out[0]=a[0][0];}',{entry:'k<4>'});assert.equal(c.metadata.workgroupStorageBytes,64);
 for(const body of ['int N=3;','N=3;'])assert.throws(()=>compile('template<int N> __global__ void k(){'+body+'}',{entry:'k<4>'}));
 for(const entry of ['k<2147483648>','k<-1>','k<1.5>','k<2,3>'])assert.throws(()=>compile(source,{entry}));
 assert.throws(()=>compile('__global__ void k(){}',{entry:'k<4>'}),/does not have/);
});
test('Missing specializations, defaults and mixed template parameter kinds are rejected',()=>{
 for(const declaration of ['template<typename T> __global__ void k(){}','template<int N=4> __global__ void k(){}','template<int N,int M> __global__ void k(){}','template<int N,typename T> __device__ void f(){} __global__ void k(){}'])assert.throws(()=>compile(declaration));
});
test('Host importer preserves template declaration and unroll directive',()=>{
 const original='#include <cuda_runtime.h>\n'+source.replace('out[0]=sum;','#pragma unroll\nfor(int j=0;j<1;j++)out[0]=sum;')+'\nint main(){}',result=kernelSource(original);assert.equal(result.functions,1);assert.match(result.source,/template <int N>/);assert.match(result.source,/#pragma unroll/);assert.equal(compile(result.source,{entry:'k<4>'}).name,'k');
});
test('Unroll hints do not affect loop semantics; other pragmas remain rejected',()=>{assert.equal(compile('__global__ void k(){\n#pragma unroll 4\nfor(int i=0;i<4;i++){}\n}').name,'k');assert.throws(()=>compile('__global__ void k(){\n#pragma pack\n}'),/directives/);});
test('Original matrixMul runs through the typed AST across multiple tiles',()=>{
 const c=compile(readFileSync('showcases/nvidia/kernels/22.cu','utf8'),{entry:'MatrixMulCUDA<16>',workgroupSize:[16,16,1]}),f=matrixFixture(16,16,32,32,16);executeCPU(c,f.buffers,f.scalars,f.groups);assert.deepEqual(f.buffers.C,f.expected);
});
