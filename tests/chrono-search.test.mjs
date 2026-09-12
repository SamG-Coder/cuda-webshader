import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Identifier aliases resolve late and through forwarding calls and conditional tests',()=>{
 const source='#define CALL TARGET\n#define TARGET twice\n#define APPLY(x) CALL(x)\n#define VALUE NUMBER\n#define NUMBER 3\n#if VALUE\n__device__ float twice(float x){return x*2.f;}\n__device__ float twice(float4 x){return x.x;}\n__global__ void k(float* out){out[0]=APPLY(VALUE);}\n#endif';
 const artifact=compile(source),out=new Float32Array(1);executeCPU(artifact,{out},{},[1]);assert.equal(out[0],6);
 const expression=compile(source.replace('#define APPLY(x) CALL(x)','#define APPLY(x) (CALL(x)+1.f)'));executeCPU(expression,{out},{},[1]);assert.equal(out[0],7);
 for(const definitions of ['#define A B\n#define B A\n','#define A B\n#define A C\n'])assert.throws(()=>compile(definitions+'__global__ void k(float* out){out[0]=A;}'),/alias|redefinition/);
 assert.throws(()=>compile('#define A f\n#define O (A(1))\n__global__ void k(float* out){out[0]=O;}'),/preprocessing/);
});
test('Only unambiguous scalar overload conversions are admitted',()=>{
 assert.throws(()=>compile('__device__ float f(float x){return x;}__device__ float f(unsigned int x){return float(x);}__global__ void k(float* out){out[0]=f(1);}'),/unambiguous/);
 assert.throws(()=>compile('__device__ __declspec(align(16)) float f(){return 1.f;}__global__ void k(){}'),/noinline/);
 assert.throws(()=>compile('__global__ void k(float* out){out[0]=(float)rint(0.5);}'),/float argument/);
});
test('The original Chrono search and distance helpers compile without rewritten bodies',()=>{
 const source=readFileSync(new URL('chrono-search.cu',import.meta.url),'utf8');
 for(const entry of ['neighborSearchNum','neighborSearchID']){const artifact=compile(source,{entry,workgroupSize:[128]});assert.match(artifact.wgsl,/cw_round_even/);assert.doesNotMatch(artifact.wgsl,/diagnostic\(off/);}
});
