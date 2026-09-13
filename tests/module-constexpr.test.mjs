import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Module constexpr arithmetic, lexical shadowing and volatile helper flags match native CUDA',()=>{
 const a=compile(readFileSync(new URL('module-constexpr.cu',import.meta.url),'utf8'),{workgroupSize:[64]}),output=new Float32Array(65*8),flags=new Uint8Array(68).fill(0xa5);
 executeCPU(a,{output,flags},{},[2]);
 const expected=readFileSync(new URL('../reports/module-constexpr-native.bin',import.meta.url));
 assert.deepEqual(Buffer.concat([Buffer.from(output.buffer),Buffer.from(flags)]),expected);
 assert.equal(a.metadata.scalars.length,0);
 assert.throws(()=>compile('constexpr int x=1;__global__ void k(){x=2;}'),/const/);
});
test('Invalid module constexpr arithmetic and declarations are rejected',()=>{
 for(const declaration of ['constexpr float x=1.f/0.f;','constexpr int x=2147483647+1;','constexpr int x=unknown;','constexpr int x=call();','constexpr int* x=0;','constexpr float x=1e100;','constexpr int x=1;constexpr int x=2;'])
  assert.throws(()=>compile(declaration+'__global__ void k(){}'));
});

test('Volatile helper reads retain atomic storage loads and reject local pointers',()=>{
 const source='__device__ bool readFlag(volatile bool* p){return *p;}__global__ void k(bool* p,uint* out){out[0]=readFlag(p);}';
 const a=compile(source,{workgroupSize:[1]});assert.match(a.wgsl,/atomicLoad/);
 assert.throws(()=>compile('__device__ void f(volatile bool* p){*p=true;}__global__ void k(){bool a=false;f(&a);}'),/pointer|storage buffers/);
});
