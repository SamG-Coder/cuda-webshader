import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';

test('Component-wise CUDA float4 helpers lower to vector arithmetic without reassociation',()=>{
  const source=`__device__ float4 accum(float4 a,float4 b,float4 c){return make_float4(a.x+truncf(b.x*c.x),a.y+truncf(b.y*c.y),a.z+truncf(b.z*c.z),a.w+truncf(b.w*c.w));}
  __global__ void sample(float4* out){out[0]=accum(make_float4(1.0f,2.0f,3.0f,4.0f),make_float4(-1.5f,1.5f,-2.5f,2.5f),make_float4(1.0f,1.0f,1.0f,1.0f));}`;
  const a=compile(source);assert.match(a.wgsl,/return \(v_a \+ trunc\(\(v_b \* v_c\)\)\);/);
  const out=new Float32Array(4);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[0,3,1,6]);
});
test('Mixed components and side effects retain scalar evaluation',()=>{
  const a=compile('__global__ void sample(float4* out){float value=1.0f;float4 v=make_float4(1.0f,2.0f,3.0f,4.0f);out[0]=make_float4(value++,value++,v.z,v.w);}');
  const out=new Float32Array(4);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[1,2,3,4]);
  assert.match(a.wgsl,/vec4<f32>\(/);
});

test('Component-wise extrema preserve lane order in vector reductions',()=>{
  const source=`__device__ float4 reduce(float4 a,float4 b){return make_float4(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z),fmaxf(a.w,b.w));}
  __global__ void sample(float4* out){out[0]=reduce(make_float4(-128.0f,-6.0f,5.0f,-21.0f),make_float4(-21.0f,2.0f,-6.0f,-128.0f));}`;
  const a=compile(source);assert.match(a.wgsl,/return max\(v_a, v_b\);/);
  const out=new Float32Array(4);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[-21,2,5,-21]);
});
