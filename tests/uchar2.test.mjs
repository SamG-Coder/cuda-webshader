import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';

test('uchar2 helpers and packed stores preserve adjacent byte pairs',()=>{
 const source='__device__ uchar2 pair(int i){uchar2 p=make_uchar2(i+257,-i);p.y+=3;return p;}__global__ void k(unsigned char*out,uint*values){int i=threadIdx.x;((uchar2*)out)[i]=pair(i);uchar2 p=((const uchar2*)out)[i];values[i]=uint(p.x)+uint(p.y)*256u+uint(sizeof(uchar2))*65536u;}';
 const c=compile(source,{workgroupSize:[32,1,1]}),out=new Uint8Array(64),values=new Uint32Array(32);executeCPU(c,{out,values},{},[1]);
 for(let i=0;i<32;i++){assert.equal(out[i*2],(i+1)&255);assert.equal(out[i*2+1],(-i+3)&255);assert.equal(values[i],out[i*2]+out[i*2+1]*256+2*65536);}
 assert.equal(c.metadata.bindings[0].stride,1);assert.equal(c.metadata.bindings[0].atomic,true);
});
test('uchar2 rejects invalid components, const writes and direct buffer layout',()=>{
 for(const body of ['uchar2 p;p.z=1;','uchar2 p=make_uchar2(1,2,3);','*((const uchar2*)out)=make_uchar2(1,2);','uchar2 p=make_uchar2(1,2);p+=p;'])assert.throws(()=>compile('__global__ void k(unsigned char*out){'+body+'}'));
 assert.throws(()=>compile('__global__ void k(uchar2*out){out[0]=make_uchar2(1,2);}'),/byte storage/);
});
