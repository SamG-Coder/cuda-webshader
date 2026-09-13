import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const declarations=`struct Pair {float x;float y;};
__device__ void operator+=(Pair& a,Pair b){a.x+=b.x;a.y+=b.y;}
__device__ void operator-=(Pair& a,float b){a.x-=b;a.y-=b;}
__device__ void operator*=(Pair& a,float b){a.x*=b;a.y*=b;}
__device__ void operator/=(Pair& a,float b){a.x/=b;a.y/=b;}
__device__ float length(Pair a){return a.x+a.y;}`;
test('Free compound operators mutate their reference and preserve value operands',()=>{
 const a=compile(declarations+`__global__ void k(float* out){Pair p;p.x=2.f;p.y=3.f;p+=p;p-=1.f;p*=4.f;p/=2.f;out[0]=p.x;out[1]=p.y;out[2]=length(p);out[3]=0.f;}`,{workgroupSize:[1]});
 const out=new Float32Array(4);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[6,10,16,0]);
});
test('Compound overloads reject const destinations and invalid signatures',()=>{
 assert.throws(()=>compile(declarations+'__global__ void k(const Pair* input){const Pair p=input[0];p+=p;}'),/const|constant/i);
 assert.throws(()=>compile('struct Pair{float x;};__device__ void operator+=(Pair a,Pair b){} __global__ void k(){}'),/mutable class reference/);
});
