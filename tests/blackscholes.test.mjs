import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {kernelSource} from '../src/sandbox/import.js';
import {seedBuffer,suggestConfig,validateConfig} from '../src/sandbox/config.js';
import {blackScholesFixture} from '../showcases/nvidia/blackscholes-fixtures.js';
test('Helper references write and forward named local scalars',()=>{
 const c=compile('__device__ void inner(float &a){a+=2.0f;} __device__ void outer(float &a,float &b){inner(a);b=a*3.0f;} __global__ void k(float* out){float a=1.0f,b=0.0f;outer(a,b);out[0]=a;out[1]=b;}',{workgroupSize:[1,1,1]});
 const out=new Float32Array(2);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[3,9]);assert.match(c.wgsl,/ptr<function, f32>/);
});
for(const [label,body] of [['alias','float a=0.0f;f(a,a);'],['storage','float a=0.0f;f(out[0],a);'],['constant','const float a=0.0f;float b=0.0f;f(a,b);'],['wrong type','int a=0;float b=0.0f;f(a,b);'],['expression','float a=0.0f;f(a+1.0f,a);']])test('Reference calls reject '+label,()=>assert.throws(()=>compile('__device__ void f(float &a,float &b){a=1.0f;b=2.0f;} __global__ void k(float* out){'+body+'}'),/Reference|reference/));
test('Reference return types, kernel parameters and vector references are rejected',()=>{
 for(const source of ['__device__ float &f(float &a){return a;} __global__ void k(){}','__global__ void k(float &a){}','__device__ void f(float2 &a){} __global__ void k(){}'])assert.throws(()=>compile(source));
});
test('Comma declarations preserve sequential initialization and loop scope',()=>{
 const c=compile('__global__ void k(int* out){int a=2,b=a+3;for(int i=0,j=3;i<j;i++){b+=i;}out[0]=b;}',{workgroupSize:[1,1,1]});const out=new Int32Array(1);executeCPU(c,{out},{},[1]);assert.equal(out[0],8);
 assert.throws(()=>compile('__global__ void k(int* out){for(int i=0,j=3;i<j;i++){}out[0]=j;}'),/Unknown identifier/);
});
test('Launch bounds are retained by host extraction and enforce maximum threads',()=>{
 const source='#include <cuda_runtime.h>\n__launch_bounds__(64) __global__ void k(float* out){out[0]=1.0f;}\nint main(){}',imported=kernelSource(source).source;
 assert.match(imported,/__launch_bounds__/);assert.equal(compile(imported,{workgroupSize:[64,1,1]}).name,'k');assert.throws(()=>compile(imported,{workgroupSize:[128,1,1]}),/Launch exceeds/);
 assert.equal(compile('__global__ __launch_bounds__(64) void k(){}',{workgroupSize:[64,1,1]}).name,'k');
 for(const bound of ['0','1025','64,2','1.5f'])assert.throws(()=>compile(`__launch_bounds__(${bound}) __global__ void k(){}`));
});
test('Fast math intrinsics validate arity',()=>{
 for(const call of ['__fdividef(1.0f)','__expf(1.0f,2.0f)','__logf()'])assert.throws(()=>compile(`__global__ void k(float* out){out[0]=${call};}`),/arguments/);
});
test('Generic buffer scaling supports positive input domains and rejects invalid values',()=>{
 const a=compile('__global__ void k(float* out){out[0]=1.0f;}'),config=suggestConfig(a),b=a.metadata.bindings[0];config.buffers.out={records:4,fill:'ramp',scale:0.5,offset:5};validateConfig(config,a.metadata);assert.deepEqual([...seedBuffer(b,config.buffers.out)],[5,5.5,6,6.5]);
 config.buffers.out.scale=Infinity;assert.throws(()=>validateConfig(config,a.metadata),/finite/);assert.throws(()=>seedBuffer(b,{records:2,fill:'one',scale:1e100}),/range/);
});
test('Original BlackScholes helpers agree with independent integrated normal distribution',()=>{
 const c=compile(readFileSync('showcases/nvidia/kernels/21.cu','utf8'),{workgroupSize:[128,1,1]}),f=blackScholesFixture(259,16);executeCPU(c,f.buffers,f.scalars,f.groups);
 for(const [name,expected]of Object.entries(f.expectedOutputs))for(let i=0;i<expected.length;i++)assert.ok(Math.abs(f.buffers[name][i]-expected[i])<=f.absoluteTolerance+f.relativeTolerance*Math.abs(expected[i]),`${name}[${i}]`);
});
