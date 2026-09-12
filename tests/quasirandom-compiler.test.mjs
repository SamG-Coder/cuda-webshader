import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Static constant matrices expose every row and constant row aliases snapshot their index',()=>{
 const a=compile('static __constant__ unsigned int table[2][3]; static __global__ void k(unsigned int*out){int row=1;unsigned int*p=&table[row][0];row=0;out[0]=p[2];out[1]=table[row][1];}'),out=new Uint32Array(2);executeCPU(a,{out},{'constant.table[1][2]':4294967295,'constant.table[0][1]':17},[1]);assert.deepEqual([...out],[4294967295,17]);
});
test('Constant table aliases reject writes, escapes, pointer updates and invalid shapes',()=>{
 for(const body of ['p[0]=1;','p++;','unsigned int*q=p;','p=&table[0][0];'])assert.throws(()=>compile('static __constant__ unsigned int table[2][3];__global__ void k(unsigned int*out){unsigned int*p=&table[1][0];'+body+'out[0]=p[0];}'));
 for(const declaration of ['table[2][200]','table[0][3]','table[2][3][4]'])assert.throws(()=>compile('static __constant__ unsigned int '+declaration+';__global__ void k(unsigned int*out){out[0]=table[0][0];}'));
});
