import test from 'node:test';import assert from 'node:assert/strict';
import {compile,parse,serializableArtifact,CompileError} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';import {packScalars,validateWorkgroup} from '../src/runtime/runtime.js';
const wrap=body=>`__global__ void test(float* output, unsigned int n) { ${body} }`;
test('AST contains real parsed index/binary/assignment nodes',()=>{const ast=parse(wrap('unsigned int i=threadIdx.x; if(i<n) output[i]=2.0f;'));assert.equal(ast.functions[0].body.body[1].yes.value.kind,'assign');});
test('Read-only inference and uniform layout are explicit',()=>{const c=compile('__global__ void k(float* inData,float* outData,float scale,int n,unsigned int shift){ outData[0]=inData[0]*scale+(float)n+(float)shift; }');assert.equal(c.metadata.bindings[0].readOnly,true);assert.equal(c.metadata.bindings[1].readOnly,false);assert.deepEqual(c.metadata.scalars.map(s=>s.offset),[0,4,8]);assert.equal(c.metadata.uniformSize,16);});
test('Parameter names that are WGSL keywords are mangled safely',()=>{assert.match(compile('__global__ void k(float* var,int fn){var[0]=(float)fn;}').wgsl,/b_var/);});
test('Conditional lowering is lazy rather than select-based',()=>{const c=compile(wrap('output[0]=n>0u?output[0]/(float)n:3.0f;'));assert.match(c.wgsl,/var cw_tmp_/);assert.doesNotMatch(c.wgsl,/select\(/);const output=new Float32Array(1);executeCPU(c,{output},{n:0},[1]);assert.equal(output[0],3);});
test('Short-circuit boolean guards do not evaluate guarded out-of-bounds loads',()=>{const c=compile(wrap('if (n>1u && output[9]>0.0f) output[0]=2.0f;'));executeCPU(c,{output:new Float32Array(1)},{n:0},[1]);});
test('Global read/write synchronization includes storageBarrier',()=>{const c=compile(wrap('output[threadIdx.x]=1.0f;__syncthreads();output[threadIdx.x]=output[0];'));assert.match(c.wgsl,/storageBarrier\(\)/);});
test('Shared-only synchronization avoids an unnecessary storage barrier',()=>{const c=compile(wrap('__shared__ float tile[128];tile[threadIdx.x]=1.0f;__syncthreads();output[threadIdx.x]=tile[0];'));assert.doesNotMatch(c.wgsl,/storageBarrier\(\)/);});
test('unsigned multiplication retains 32-bit wrap semantics in oracle',()=>{const c=compile('__global__ void k(unsigned int* outData){outData[0]=4294967295u*4294967295u;}');const outData=new Uint32Array(1);executeCPU(c,{outData},{},[1]);assert.equal(outData[0],1);});
test('Fixed numeric defines can be specialized at compile time',()=>{const c=compile('#define SIZE 64\n__global__ void k(float* x){__shared__ float a[SIZE];a[threadIdx.x]=0.0f;__syncthreads();x[threadIdx.x]=a[threadIdx.x];}',{defines:{SIZE:128}});assert.equal(c.metadata.workgroupStorageBytes,512);});
test('C integer index casts and shifts have explicit WGSL types',()=>{const c=compile(wrap('int i=(int)threadIdx.x; i>>=1; output[i]=(float)i;'));assert.match(c.wgsl,/i32\(cw_thread.x\)/);assert.match(c.wgsl,/u32\(1i\)/);});
test('Multiple kernels require an explicit entry selection',()=>{const src='__global__ void a(){} __global__ void b(){}';assert.throws(()=>compile(src),/Multiple kernels/);assert.equal(compile(src,{entry:'b'}).name,'b');});
test('Serialized artifacts contain no AST and roundtrip as JSON',()=>{const s=serializableArtifact(compile(wrap('output[0]=(float)n;')));assert.equal(s.ast,undefined);assert.equal(JSON.parse(JSON.stringify(s)).wgsl,s.wgsl);});
test('Uniform packing uses f32/i32/u32 little-endian fields',()=>{const c=compile('__global__ void k(float* o,float a,int b,unsigned int c){o[0]=a+(float)b+(float)c;}');const p=packScalars(c.metadata,{a:1.25,b:-2,c:4294967295});const d=new DataView(p);assert.equal(d.getFloat32(0,true),1.25);assert.equal(d.getInt32(4,true),-2);assert.equal(d.getUint32(8,true),4294967295);});
for(const values of [{n:-1},{n:1.5},{n:4294967296},{n:NaN},{n:Infinity},{n:'12'},{extra:1,n:2},{}])test(`Uniform validation rejects ${JSON.stringify(values)}`,()=>{assert.throws(()=>packScalars(compile(wrap('output[0]=(float)n;')).metadata,values));});
const bad=[
 ['double pointer','__global__ void k(double* x){}',/Unsupported type/],
 ['float3 ABI','__global__ void k(float3* x){}',/incompatible/],
 ['bool ABI','__global__ void k(bool* x){}',/incompatible/],
 ['double literal',wrap('output[0]=0.5;'),/suffix/],
 ['float atomics',wrap('atomicAdd(&output[0],1.0f);'),/integer atomics/],
 ['dynamic shared',wrap('extern __shared__ float a[];'),/sharedMemoryBytes/],
 ['local pointer',wrap('float a[4];float* x=a;'),/Local pointers/],
 ['unexpanded include','#include <cuda.h>\n__global__ void k(){}',/preprocess/],
 ['function macro','#define F(x) x\n__global__ void k(){}',/numeric/],
 ['CUDA allocation',wrap('cudaMalloc(output,n);'),/Unsupported function/],
 ['warp shuffle',wrap('output[0]=__shfl_sync(0xffffffffu,1.0f,0);'),/Unsupported function/],
 ['const write','__global__ void k(const float* x){x[0]=1.0f;}',/const/],
 ['pointer arithmetic',wrap('output=output+1;'),/scalar|operand/],
 ['post-increment in expression',wrap('int i=0;output[i++]=1.0f;'),/inside expressions/],
 ['unknown identifier',wrap('output[0]=missing;'),/Unknown identifier/],
 ['recursion','__device__ float f(float x){return f(x);} __global__ void k(float* x){x[0]=f(x[0]);}',/Recursive/],
 ['helper capture','__device__ float f(float a){return output[0];} __global__ void k(float* output){output[0]=f(1.0f);}',/Unknown identifier/],
 ['unsupported vector remainder',wrap('float4 p=make_float4(1.0f,2.0f,3.0f,4.0f);p=p%1.0f;'),/Vector arithmetic/],
 ['invalid break',wrap('break;'),/only valid/],
 ['void local',wrap('void x;'),/void type/],
 ['scalar parameter mutation',wrap('n=3u;'),/read-only/],
 ['malformed comment','/* never closed',/Unclosed/],
 ['malformed block','__global__ void k(){',/Unclosed/],
 ['large literal',wrap('int x=999999999999;'),/32-bit/]
];
for(const [name,source,pattern]of bad)test(`Reject unsupported/unsafe source: ${name}`,()=>assert.throws(()=>compile(source),pattern));
test('Errors carry source line and column',()=>{try{compile('__global__ void k() {\n  float x=unknown;\n}');assert.fail();}catch(e){assert.ok(e instanceof CompileError);assert.equal(e.line,2);assert.ok(e.message.includes('^'));}});
test('Device workgroup/storage limits are enforced before pipeline creation',()=>{
 const c=compile(wrap('__shared__ float a[128];a[threadIdx.x]=0.0f;__syncthreads();output[threadIdx.x]=a[0];'));
 const limits={maxComputeWorkgroupSizeX:256,maxComputeWorkgroupSizeY:256,maxComputeWorkgroupSizeZ:64,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupStorageSize:16384,maxStorageBuffersPerShaderStage:8};
 assert.doesNotThrow(()=>validateWorkgroup(c.metadata,limits));assert.throws(()=>validateWorkgroup(c.metadata,{...limits,maxComputeInvocationsPerWorkgroup:64}));assert.throws(()=>validateWorkgroup(c.metadata,{...limits,maxComputeWorkgroupStorageSize:256}));
});
test('Hexadecimal literals ending in f are integer digits, not float suffixes',()=>{
 const c=compile('__global__ void k(unsigned int* output){output[0]=0xff;output[1]=0xDEADBEEF;output[2]=0xFFu;}');const output=new Uint32Array(3);executeCPU(c,{output},{},[1]);assert.deepEqual(Array.from(output),[255,0xdeadbeef,255]);assert.match(c.wgsl,/255i/);
});
test('Signed right shift remains signed with an unsigned shift-count operand',()=>{
 const c=compile('__global__ void k(int* output){int a=-8;output[0]=a>>1u;a>>=1u;output[1]=a;}');const output=new Int32Array(2);executeCPU(c,{output},{},[1]);assert.deepEqual(Array.from(output),[-4,-4]);assert.doesNotMatch(c.wgsl,/u32\(v_a\)/);
});
test('Failed uniform updates leave the previous byte snapshot unchanged',()=>{
 const c=compile('__global__ void k(float a,unsigned int n){}');const target=packScalars(c.metadata,{a:1,n:2}),before=new Uint8Array(target).slice();assert.throws(()=>packScalars(c.metadata,{a:99,n:-1},target));assert.deepEqual(new Uint8Array(target),before);
});
