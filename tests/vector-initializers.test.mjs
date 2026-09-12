import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Vector aggregate initialization supports full, partial, empty and trailing-comma lists',()=>{
 const c=compile('__global__ void k(float4* out){float3 acc={0.0f,0.0f,0.0f};float4 a={1.0f,2.0f,3.0f,4.0f},b={5.0f,},z={};out[0]=a;out[1]=b;out[2]=z;out[3]=make_float4(acc.x,acc.y,acc.z,1.0f);}',{workgroupSize:[1]});
 const out=new Float32Array(16);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[1,2,3,4,5,0,0,0,0,0,0,0,0,0,0,1]);
});
test('Vector aggregate initializers resolve dependent trait types in helpers',()=>{
 const c=compile('template<class T> struct vec3 {typedef float3 Type;}; template<> struct vec3<float>{typedef float3 Type;};template<class T> __device__ T f(T x){typename vec3<T>::Type a={x,2.0f,3.0f};return a.x+a.y+a.z;}__global__ void k(float* out){out[0]=f(4.0f);}',{workgroupSize:[1]});
 const out=new Float32Array(1);executeCPU(c,{out},{},[1]);assert.equal(out[0],9);
});
test('Vector aggregate initializers reject excess elements, implicit conversions and unsupported aggregates',()=>{
 for(const declaration of ['float2 a={1.0f,2.0f,3.0f};','float2 a={1,2};','int2 a={1.0f};','float a={1.0f};','float a[2]={1.0f,2.0f};','__shared__ float2 a={};'])assert.throws(()=>compile('__global__ void k(){'+declaration+'}'),/initializer|initializers/i);
});
