import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {packScalars} from '../src/runtime/runtime.js';
const source=readFileSync(new URL('chrono-counter-time.cu',import.meta.url),'utf8');
const cases=JSON.parse(readFileSync(new URL('../reports/chrono-counter-time-native.json',import.meta.url)));
test('Chrono Counters and double launch times preserve native 64-bit boundary cases',()=>{
 const artifact=compile(source,{workgroupSize:[1]});
 assert.equal(artifact.metadata.scalars.length,46);
 for(const {scalars,output} of cases){
  assert.doesNotThrow(()=>packScalars(artifact.metadata,scalars));
  const actual=new Uint32Array(96);executeCPU(artifact,{output:actual},scalars,[1]);
  assert.deepEqual([...actual],output);
 }
});
test('Double launch words reject missing and out-of-range input transactionally',()=>{
 const a=compile('__global__ void k(float* o,double time){o[0]=(float)time;}');
 const target=new ArrayBuffer(a.metadata.uniformSize);new Uint8Array(target).fill(0xa5);
 for(const values of [{'time.lo':0},{'time.lo':-1,'time.hi':0},{'time.lo':0,'time.hi':4294967296}]){
  assert.throws(()=>packScalars(a.metadata,values,target));
  assert.ok(new Uint8Array(target).every(v=>v===0xa5));
 }
 for(const s of ['__global__ void k(double* p){}','__global__ void k(double& p){}','__global__ void k(double p=0.0){}','__global__ void k(){double p[2];}','struct C {size_t n;};__global__ void k(C* c){c[0].n=1;}'])assert.throws(()=>compile(s));
});
test('Binary64 constant records remain numbers in the independent CPU oracle',()=>{
 const a=compile('struct P {double x;};__constant__ P p;__global__ void k(float* o){o[0]=(float)(p.x-1.0);}',{workgroupSize:[1]});
 const bits=new DataView(new ArrayBuffer(8));bits.setFloat64(0,1+2**-40,true);
 const o=new Float32Array(1);executeCPU(a,{o},{'constant.p.x.lo':bits.getUint32(0,true),'constant.p.x.hi':bits.getUint32(4,true)},[1]);
 assert.equal(o[0],2**-40);
});
