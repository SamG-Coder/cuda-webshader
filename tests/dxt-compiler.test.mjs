import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Local scalar pointers forward through typed helper chains without losing writeback',()=>{for(const type of ['float','int','uint','short','ushort']){const a=compile(`__device__ void leaf(${type}* p){*p=7;} __device__ void middle(${type}* p){leaf(p);} __device__ void top(${type}* p){middle(p);} __global__ void k(int*out){${type} value=1;top(&value);out[0]=value;}`),out=new Int32Array(1);executeCPU(a,{out},{},[1]);assert.equal(out[0],7);}});
test('Local pointer forwarding rejects arithmetic, escapes, const writes, mismatched types and aliases',()=>{for(const source of [
 '__device__ void leaf(float*p){*p=1;} __device__ void mid(float*p){leaf(p+1);}',
 '__device__ void leaf(int*p){*p=1;} __device__ void mid(float*p){leaf(p);}',
 '__device__ void mid(float*p){float* q=p;}',
 '__device__ void leaf(float*a,float*b){*a=1;*b=2;} __device__ void mid(float*p){leaf(p,p);}',
 '__device__ void leaf(float*p){*p=1;} __device__ void mid(const float*p){leaf(p);}'
 ])assert.throws(()=>compile(source+' __global__ void k(float*out){float x=0;mid(&x);out[0]=x;}'));});
test('Constant vectors retain initialized components and zero-fill omitted components',()=>{for(const [type,Type,values]of [['float',Float32Array,[1.25,-2]],['int',Int32Array,[-17,19]],['uint',Uint32Array,[17,19]]]){const a=compile(`__constant__ ${type}3 metric={${values.join(',')}}; __global__ void k(${type}*out){out[0]=metric.x;out[1]=metric.y;out[2]=metric.z;}`),out=new Type(3);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[...values,0]);}});
test('Constant vector initializers reject excess components and values outside scalar ranges',()=>{for(const declaration of ['float2 x={1,2,3}','uint2 x={-1,2}','int2 x={2147483648,0}','float2 x={1e100,0}','float2 x={sin(1),0}'])assert.throws(()=>compile('__constant__ '+declaration+'; __global__ void k(float*out){out[0]=x.x;}'));});
test('rintf uses ties-to-even and preserves negative zero',()=>{const a=compile('__global__ void k(float*input,float*out){unsigned i=blockIdx.x;out[i]=rintf(input[i]);}'),input=Float32Array.from([-3.5,-2.5,-1.5,-.5,-.25,-0,0,.25,.5,1.5,2.5,3.5]),out=new Float32Array(input.length);executeCPU(a,{input,out},{},[input.length]);assert.deepEqual([...out],[-4,-2,-2,-0,-0,-0,0,0,0,2,2,4]);});

test('Large float literals preserve scientific notation in WGSL',()=>{for(const value of ['3.402823466e38f','1e21f','-1e30f']){const a=compile(`__global__ void k(float*out){out[0]=${value};}`);assert.doesNotMatch(a.wgsl,/e[+-]?\d+\.0f/i);assert.match(a.wgsl,/e[+-]?\d+f/i);}});

test('First-tile lowering validates its parent and synchronization handle',()=>{
 const source=(parent='cta',handle='tile')=>`namespace cg=cooperative_groups; __global__ void k(float*out){cg::thread_block cta=cg::this_thread_block();const int lane=threadIdx.x;cg::thread_group tile=cg::tiled_partition(${parent},16);if(lane<16){out[lane]=1;cg::sync(tile);cg::sync(${handle});}}`;
 assert.equal(compile(source(),{workgroupSize:[64,1,1]}).metadata.tiledGroups,'predicated-first-tile');
 assert.throws(()=>compile(source('missing'),{workgroupSize:[64,1,1]}),/existing thread block/);
 assert.throws(()=>compile(source('cta','missing'),{workgroupSize:[64,1,1]}),/group handle/);
 for(const workgroupSize of [[24,1,1],[16,2,1]])assert.throws(()=>compile(source(),{workgroupSize}),/one-dimensional block/);
});

test('Predicated tile groups reject handle escapes and varying loop induction',()=>{
 const kernel=body=>`namespace cg=cooperative_groups;__device__ void stage(float*out,cg::thread_group tile){${body}} __global__ void k(float*out){cg::thread_block cta=cg::this_thread_block();const int lane=threadIdx.x;cg::thread_group tile=cg::tiled_partition(cta,16);if(lane<16){stage(out,tile);}}`;
 for(const body of ['bool x=tile;','for(int i=0;i<4;i++){i+=threadIdx.x;cg::sync(tile);}','if(threadIdx.x<4)return;cg::sync(tile);'])assert.throws(()=>compile(kernel(body),{workgroupSize:[64,1,1]}));
});

test('Fixed array helper parameters preserve pointer reads and reject invalid dimensions',()=>{
 const a=compile('__device__ float sum(const float p[2]){return p[0]+p[1];} __global__ void k(float*input,float*out){out[0]=sum(input);}'),input=Float32Array.of(3,7),out=new Float32Array(1);executeCPU(a,{input,out},{},[1]);assert.equal(out[0],10);
 for(const dimension of ['0','-1','65537','2][2'])assert.throws(()=>compile(`__device__ float sum(float p[${dimension}]){return p[0];} __global__ void k(float*input,float*out){out[0]=sum(input);}`));
});
