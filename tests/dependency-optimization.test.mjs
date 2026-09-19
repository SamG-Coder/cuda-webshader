import test from 'node:test';
import assert from 'node:assert/strict';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';
import {trimWgslDependencies} from '../src/compiler/wgsl-dependencies.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {optimizerCases} from './dependency-optimization-cases.js';
const metadata=c=>{const {optimization,...abi}=c.metadata;return abi;};
for(const fixture of optimizerCases)test(`Dependency optimizer: ${fixture.name}`,()=>{
 const baseline=compile(fixture.source,{optimize:false,workgroupSize:[32]}),trimmed=compile(fixture.source,{workgroupSize:[32]}),optimized=compile(fixture.source,{optimize:'specialize',workgroupSize:[32]});
 assert.deepEqual(metadata(trimmed),metadata(baseline));assert.deepEqual(metadata(optimized),metadata(baseline));
 for(const input of [-3,0,1,2,7,19]){
  const outputs=[baseline,trimmed,optimized].map(c=>{const output=new Int32Array(32);executeCPU(c,{output},{input},[1]);return output;});
  assert.deepEqual(outputs[1],outputs[0]);assert.deepEqual(outputs[2],outputs[0]);
 }
 const report=optimized.metadata.optimization.specialization;
 if(fixture.prunes){assert.ok(report.branchesPruned+report.conditionalExpressionsPruned>0);assert.equal(report.fallback,undefined);}
});
test('Prunes transitive unreachable helpers, retaining resource ABI and exact live functions',()=>{
 const source='__device__ int unusedLeaf(int n){return n*2;} __device__ int unused(int n){return unusedLeaf(n);} __device__ int used(int n){return n+1;} __global__ void k(int* output,int unusedUniform){output[0]=used(1);}';
 const a=compile(source,{optimize:false}),b=compile(source);
 assert.doesNotMatch(b.wgsl,/fn f_unused/);assert.match(b.wgsl,/fn f_used/);
 assert.deepEqual(metadata(b),metadata(a));assert.equal(b.metadata.optimization.baselineBytes,Buffer.byteLength(a.wgsl));
 assert.deepEqual(b.metadata.optimization.removedFunctions,['f_unusedLeaf','f_unused']);
 assert.equal(serializableArtifact(b).metadata.optimization,b.metadata.optimization);
});
test('Keeps all WGSL stage entries, nested comments cannot introduce dependencies',()=>{
 const text=`/* fn fake() { /* nested */ } */\nfn unused()->f32{return 2.;}\nfn helper()->f32{return 1.;}\n@compute @workgroup_size(1) fn main(){let x=helper();}\n@compute @workgroup_size(2) fn other(){let y=helper();}`;
 const result=trimWgslDependencies(text);
 assert.match(result.wgsl,/@compute @workgroup_size\(2\) fn other/);assert.match(result.wgsl,/fn helper/);
 assert.deepEqual(result.report.removedFunctions,['unused']);
 assert.equal(trimWgslDependencies(result.wgsl).wgsl,result.wgsl);
});
test('WGSL pass preserves non-function declarations and module roots',()=>{
 const text='const declared=helper();\nfn helper()->i32{return 1;}\n@group(0) @binding(0) var<storage,read_write> output:array<i32>;\n@compute @workgroup_size(1) fn main(){output[0]=declared;}';
 assert.equal(trimWgslDependencies(text).wgsl,text);
});
for(const text of ['/* broken','fn main(){','fn other(){}','fn main(){} fn main(){}'])test(`WGSL fail-closed ${text}`,()=>{
 assert.equal(trimWgslDependencies(text).wgsl,text);assert.ok(trimWgslDependencies(text).report.skipped);
});
test('Original validation remains authoritative even in pruned paths',()=>{
 const source='__device__ int helper(int mode){if(mode==1)return 1;return not_supported();} __global__ void k(int* o){o[0]=helper(1);}';
 for(const optimize of [false,'dependencies','specialize'])assert.throws(()=>compile(source,{optimize}),/Unsupported function/);
});
test('Recursion is not hidden by specialization or final WGSL trimming',()=>{
 const s='__device__ int f(int mode){if(mode==1)return 1;return f(mode);} __global__ void k(int* o){o[0]=f(1);}';
 assert.throws(()=>compile(s,{optimize:'specialize'}),/Recursive/);
});
test('Entry-root analysis does not union modes from other kernels',()=>{
 const s=optimizerCases[0].source+'\n__global__ void other(int* o){Sink s=newSink(0);s=emit(s,3);o[0]=s.count;}';
 const c=compile(s,{entry:'k',optimize:'specialize'});
 assert.ok(c.metadata.optimization.specialization.recordFields.some(f=>f.field==='mode'&&f.value===1));
});
test('Externally supplied record fields stay dynamic',()=>{
 const s='struct S{int mode;};__device__ S make(){S s;s.mode=1;return s;}__device__ int f(S s){if(s.mode==1)return 7;return -1;}__global__ void k(int* output,S s){output[0]=f(s);}';
 const c=compile(s,{optimize:'specialize'});
 assert.equal(c.metadata.optimization.specialization.branchesPruned,0);
});
test('Function specialization does not evaluate signed overflow or shifts',()=>{
 const s='__device__ int f(int n){if(n+1<0)return 7;if((n>>1)==0)return 9;return 3;} __global__ void k(int* o){o[0]=f(2147483647);}';
 const c=compile(s,{optimize:'specialize'});assert.equal(c.metadata.optimization.specialization.branchesPruned,0);
});
test('Invalid optimizer option is a useful diagnostic',()=>assert.throws(()=>compile('__global__ void k(){}',{optimize:'maximum'}),/optimize must/));
