import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {scanUpdateFixture} from '../showcases/nvidia/scan-update-fixtures.js';
for(const [type,Type,values]of [['uint',Uint32Array,[16777217,2147483648,4294967295,0]],['int',Int32Array,[16777217,-2147483648,2147483647,-1]]])for(const size of [2,4])test(`${type}${size} keeps integer precision through helpers and component writes`,()=>{
 const source=`__device__ ${type}${size} copy(${type}${size} a){return a;} __global__ void k(${type}${size}* out){${type}${size} a=copy(make_${type}${size}(${values.slice(0,size).map(v=>type==='uint'?v+'u':v===-2147483648?'(-2147483647-1)':v).join(',')}));a.x+=1;out[0]=a;}`;
 const c=compile(source,{workgroupSize:[1,1,1]}),out=new Type(size);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],values.slice(0,size).map((v,i)=>i?v:v+1));assert.equal(c.metadata.bindings[0].stride,size*4);assert.match(c.wgsl,new RegExp(`vec${size}<${type==='uint'?'u':'i'}32>`));
});
test('Integer vector layouts and constructors reject incompatible usage',()=>{
 for(const type of ['int3','uint3'])assert.throws(()=>compile(`__global__ void k(${type}* out){}`),/incompatible/);
 assert.throws(()=>compile('__global__ void k(uint4* out){out[0]=make_uint4(1u,2u);}'),/arguments/);
 assert.throws(()=>compile('__global__ void k(uint4* out){out[0]=make_float4(1.0f,2.0f,3.0f,4.0f);}'),/convert/);
 assert.throws(()=>compile('__global__ void k(uint2* out){out[0].z=1u;}'),/components/);
});
test('Shared scalar publication and atomics synchronize correctly',()=>{
 const c=compile('__global__ void k(unsigned int* out){__shared__ unsigned int count;if(threadIdx.x==0)count=0u;__syncthreads();atomicAdd(&count,1u);__syncthreads();out[threadIdx.x]=count;}',{workgroupSize:[32,1,1]}),out=new Uint32Array(32);executeCPU(c,{out},{},[1]);assert.ok(out.every(v=>v===32));assert.match(c.wgsl,/var<workgroup> s_count: atomic<u32>/);assert.match(c.wgsl,/atomicLoad\(&s_count\)/);
 assert.throws(()=>compile('__global__ void k(){__shared__ uint count=1u;}'),/initializer/);
});
test('Original uniformUpdate preserves uint4 precision and wraps at 32 bits',()=>{const c=compile(readFileSync('showcases/nvidia/kernels/24.cu','utf8'),{workgroupSize:[256,1,1]}),f=scanUpdateFixture(4,256,16);executeCPU(c,f.buffers,{},f.groups);assert.deepEqual(f.buffers.d_Data,f.expected);});
