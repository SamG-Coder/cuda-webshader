import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=readFileSync(new URL('./smoke-integration-kernel.cuh',import.meta.url),'utf8');
test('Smoke integration keeps four independent tuple bindings and original texture state',()=>{
 const m=compile(source,{entry:'integrate_functor'}).metadata;
 assert.deepEqual(m.bindings.map(b=>[b.elementType,b.readOnly]),[['vec4<f32>',false],['vec4<f32>',false],['vec4<f32>',true],['vec4<f32>',true]]);
 assert.equal(m.textures[0].format,'rgba32float');assert.equal(m.textures[0].dimension,'3d');
});
test('Original smoke depth functor projects float4 positions into scalar keys with partial bounds',()=>{
 const a=compile(source,{entry:'calcDepth_functor',workgroupSize:[32,1,1]}),tuple0=Float32Array.from({length:20*4},(_,i)=>i*.25),tuple1=new Float32Array(20).fill(-77),original=tuple0.slice();
 executeCPU(a,{tuple0,tuple1},{'sortVector.x':.25,'sortVector.y':-.5,'sortVector.z':1,count:17},[1]);
 for(let i=0;i<20;i++)assert.equal(tuple1[i],i<17?-(tuple0[i*4]*.25-tuple0[i*4+1]*.5+tuple0[i*4+2]):-77);
 assert.deepEqual(tuple0,original);
});
const generic='struct transform {float scale;float bias;__host__ __device__ transform(float b,float a):bias(b),scale(a){}template<class Tuple> __device__ void operator()(Tuple t){float v=cuda::std::get<0>(t);float result=v*scale+bias;cuda::std::get<1>(t)=result;}};';
test('Typed zip state and tuple inference do not depend on NVIDIA names or field order',()=>{
 const a=compile(generic,{entry:'transform',workgroupSize:[4,1,1]}),tuple0=Float32Array.of(1,2,3,4),tuple1=new Float32Array(4).fill(-77);
 executeCPU(a,{tuple0,tuple1},{scale:3,bias:2,count:3},[1]);assert.deepEqual([...tuple1],[5,8,11,-77]);
});
test('Zip lowering rejects ambiguous tuple types, gaps and nontrivial state construction',()=>{
 for(const s of [
  generic.replace('scale(a)','scale(b)'),
  generic.replace('get<1>','get<3>'),
  generic.replace('get<1>','get<4>'),
  generic.replace('float result=v*scale+bias;','float4 result=make_float4(v);').replace('get<1>','get<0>'),
  generic.replace('float result=v*scale+bias;','unsigned tuple2=0;float result=v*scale+bias;'),
  generic.replace('float scale;','float* scale;')
 ])assert.throws(()=>compile(s,{entry:'transform'}));
});
