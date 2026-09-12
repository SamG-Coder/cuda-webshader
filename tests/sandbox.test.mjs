import test from 'node:test';
import assert from 'node:assert/strict';
import {kernelSource} from '../src/sandbox/import.js';
import {compile} from '../src/compiler/compiler.js';
import {suggestConfig,validateConfig} from '../src/sandbox/config.js';
test('Desktop import preserves device helpers, numeric macros and diagnostic line positions',()=>{
 const source='#include <cuda_runtime.h>\n#define COUNT 16\n// pretend __global__ void ignored() {}\n__device__ float twice(float x) { return x*2.0f; }\n__global__ void fill(float* output) { output[threadIdx.x]=twice(1.0f); }\nint main(){ fill<<<1,COUNT>>>(nullptr); }';
 const result=kernelSource(source);assert.equal(result.extracted,true);assert.equal(result.functions,2);assert.equal(result.source.split('\n')[4],source.split('\n')[4]);assert.equal(compile(result.source,{entry:'fill'}).name,'fill');assert.ok(!result.source.includes('nullptr'));
});
test('Sandbox budgets reject excessive dispatches and storage before GPU allocation',()=>{
 const artifact=compile('__global__ void fill(float* output,unsigned int n){unsigned int i=threadIdx.x;if(i<n)output[i]=1.0f;}');const config=suggestConfig(artifact);assert.ok(validateConfig(config,artifact.metadata)>0);
 assert.throws(()=>validateConfig({...config,groups:[65535,65535,1]},artifact.metadata),/invocations/);
 config.buffers.output.records=1048577;assert.throws(()=>validateConfig(config,artifact.metadata),/records/);
});
test('Kernel-only source stays unchanged and host-only files fail explicitly',()=>{
 const source='__global__ void f(float* out) { out[0]=1.0f; }';assert.equal(kernelSource(source).source,source);assert.throws(()=>kernelSource('#include <cuda_runtime.h>\nint main() {}'),/No standalone/);
});
test('Windows desktop CUDA files accept numeric macros with trailing comments',()=>{
 const source='#include <cuda_runtime.h>\r\n#define REFRESH_DELAY 10 // ms\r\n__global__ void f(float* out) { out[0]=(float)REFRESH_DELAY; }\r\nint main() { return 0; }\r\n';
 assert.equal(compile(kernelSource(source).source,{entry:'f'}).name,'f');
});

test('Desktop import keeps referenced classes and transitive records without orphan methods',()=>{
 const source='#include <cuda_runtime.h>\nclass Inner { public: float x; __host__ __device__ Inner():x(2.f){} };\nclass Outer { public: Inner inner; __host__ __device__ float get() const {return inner.x;} };\nclass HostOnly { FILE* file; __device__ float ignored(){return 0.f;} };\n__global__ void k(float* out){Outer value;out[0]=value.get();}\nint main(){return 0;}';
 const imported=kernelSource(source).source;assert.ok(imported.includes('class Inner'));assert.ok(imported.includes('class Outer'));assert.ok(!imported.includes('HostOnly'));assert.ok(!imported.includes('ignored'));
 assert.equal(imported.split('\n').length,source.split('\n').length);assert.equal(compile(imported,{entry:'k'}).name,'k');
});
