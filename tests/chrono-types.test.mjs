import test from 'node:test';import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';import {packScalars} from '../src/runtime/runtime.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Scoped enum values retain integer representation and scope',()=>{
 const a=compile('enum class A { ZERO, NEG=-7, NEXT, FLAG=(1<<5)|2, COPY=A::FLAG }; enum class B : unsigned int { ZERO=0xffffffffu }; __global__ void k(int* a,unsigned int* b){A v=A::COPY;a[0]=(int)v;a[1]=(int)A::NEXT;b[0]=(unsigned int)B::ZERO;}');
 assert.match(a.wgsl,/34/);assert.match(a.wgsl,/-6/);assert.match(a.wgsl,/4294967295u/);
 const mixed=compile('enum class E { X=~0u>>1 }; enum class U:unsigned int { X=0xffffffffu+1u };__global__ void k(unsigned int* out){out[0]=(unsigned int)E::X;out[1]=(unsigned int)U::X;}');assert.match(mixed.wgsl,/2147483647/);
 const values=new Uint32Array(2);executeCPU(mixed,{out:values},{},[1]);assert.deepEqual([...values],[2147483647,0]);
 for(const s of ['enum class A { X,X };','enum class float { X };','enum class A {X=2147483648};','enum class A {X=1/0};','enum class A {X=1<<32};','enum class A : float {X};','enum class A { X=f() };'])assert.throws(()=>compile(s+'__global__ void k(){}'));
 assert.throws(()=>compile('enum class A {X};__global__ void k(int* a){a[0]=(int)A::Y;}'),/Unknown scoped enum member/);
 assert.throws(()=>compile('enum class A {X};__global__ void k(int* a){a[0]=X;}'),/Unknown/);
});
test('Large constant records carry boolean flags as validated integer uniforms',()=>{
 const fields=Array.from({length:120},(_,i)=>'float f'+i+';').join('');
 const a=compile('struct P {'+fields+'bool periodic;};__constant__ P paramsD;__global__ void k(float* out){out[0]=paramsD.periodic?paramsD.f119:paramsD.f0;}');
 const flag=a.metadata.scalars.find(s=>s.name==='constant.paramsD.periodic');assert.equal(flag.type,'u32');assert.equal(flag.sourceType,'bool');
 const data=packScalars(a.metadata,{'constant.paramsD.periodic':true});assert.equal(new DataView(data).getUint32(flag.offset,true),1);
 const original=new Uint8Array(data).slice();assert.throws(()=>packScalars(a.metadata,{'constant.paramsD.periodic':2},data),/boolean/);assert.deepEqual(new Uint8Array(data),original);
 assert.throws(()=>compile('struct P {'+Array.from({length:257},(_,i)=>'float f'+i+';').join('')+'};__global__ void k(){}'),/256 fields/);
});
