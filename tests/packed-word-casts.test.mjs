import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';

test('Byte-buffer word loads populate local uchar4 without numeric conversion',()=>{
  const source='__global__ void k(const unsigned char*input,uint*out){int i=threadIdx.x;const unsigned char*p=input+i*8;uchar4 pixel;*(uint*)&pixel=((const uint*)p)[1];out[i*2]=*(uint*)&pixel;pixel.z=17;out[i*2+1]=*(uint*)&pixel;}';
  const c=compile(source,{workgroupSize:[8,1,1]}),input=Uint8Array.from({length:64},(_,i)=>(i*37+201)&255),out=new Uint32Array(16);
  executeCPU(c,{input,out},{},[1]);const view=new DataView(input.buffer);
  for(let i=0;i<8;i++){const expected=view.getUint32(i*8+4,true);assert.equal(out[i*2],expected);assert.equal(out[i*2+1],((expected&0xff00ffff)|(17<<16))>>>0);}
  assert.equal(c.metadata.bindings[0].readOnly,true);
});
test('Packed word views preserve const and reject unsupported reinterpretation',()=>{
  for(const body of ['const uchar4 p=make_uchar4(1,2,3,4);*(uint*)&p=7u;','float p=1.0f;*(uint*)&p=7u;','uchar4 p;((uint*)&p)[1]=7u;','uchar4 p;*(const uint*)&p=7u;','*(uint*)input=7u;'])assert.throws(()=>compile('__global__ void k(unsigned char*input){'+body+'}'));
});
test('Vector pointer views store consecutive scalar records with alias offsets',()=>{
  const c=compile('__global__ void k(float*out){int i=threadIdx.x;float*p=out+4;((float4*)p)[i]=make_float4(float(i),2.0f,3.0f,4.0f);float4 v=((const float4*)p)[i];out[i+20]=v.z;}',{workgroupSize:[4,1,1]}),out=new Float32Array(24);
  executeCPU(c,{out},{},[1]);assert.deepEqual([...out.slice(0,4)],[0,0,0,0]);for(let i=0;i<4;i++)assert.deepEqual([...out.slice(4+i*4,8+i*4)],[i,2,3,4]);assert.deepEqual([...out.slice(20)],[3,3,3,3]);assert.equal(c.metadata.bindings[0].readOnly,false);
  for(const source of ['__global__ void k(const float*out){*((float4*)out)=make_float4(1,2,3,4);}','__global__ void k(float*out){*((uint4*)out)=make_uint4(1,2,3,4);}'])assert.throws(()=>compile(source));
});
test('Scalar components of vector pointer views remain writable storage elements',()=>{
  const c=compile('__global__ void k(float*out){((float4*)out)[0].z=7.0f;((float4*)out)[0].x++;}',{workgroupSize:[1,1,1]}),out=Float32Array.of(1,2,3,4);
  executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[2,2,7,4]);assert.doesNotMatch(c.wgsl,/vec4<f32>\([^;]+\)\.[xz]\s*=/);
});
