import test from 'node:test';import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Double expression promotions preserve cancellation and compound assignment',()=>{
 const source=readFileSync(new URL('./float64-expression-kernel.cuh',import.meta.url),'utf8');
 const artifact=compile(source,{entry:'expressions',workgroupSize:[1]});
 const out=new Float32Array(6);executeCPU(artifact,{a:new Float32Array([16777216]),b:new Float32Array([1]),out},{count:1},[1]);
 assert.deepEqual([...out],[16777216,16777216,16777216,1,16777216,1]);
 assert.match(artifact.wgsl,/cw_d_to_f32\(cw_d_add/);
});
test('Double comparisons evaluate side-effecting operands once',()=>{
 const source='__device__ float next(float& x){x+=1.f;return x;}__global__ void k(float* out){float x=1.f;out[0]=(next(x)+0.0)<=3.0;out[1]=x;out[2]=-(0.1+0.2);out[3]=(0.0/0.0)<=1.0;}';
 const artifact=compile(source,{workgroupSize:[1]}),out=new Float32Array(4);executeCPU(artifact,{out},{},[1]);
 assert.deepEqual([...out],[1,2,Math.fround(-.3),0]);assert.match(artifact.wgsl,/cw_d_le/);
});
test('Extent addition, subtraction and float conversion preserve unsigned size semantics',()=>{
 const artifact=compile('__global__ void k(cudaExtent size,float* out,uint* low){out[0]=float(size.width-1);out[1]=float(size.width+1);low[0]=uint(size.width-1);}',{workgroupSize:[1]}),out=new Float32Array(2),low=new Uint32Array(1);
 executeCPU(artifact,{out,low},{'size.width':0,'size.height':1,'size.depth':1},[1]);assert.deepEqual([...out],[Math.fround(Number((1n<<64n)-1n)),1]);assert.equal(low[0],0xffffffff);
});
test('Unsupported double storage, integer conversions and double transcendental calls stay explicit',()=>{
 for(const source of ['__global__ void k(){double x[2];}','__global__ void k(int* o){o[0]=1.5;}','__global__ void k(float* o){o[0]=exp(2.0);}','__global__ void k(float* o){o[0]=0.5%0.25;}'])assert.throws(()=>compile(source));
});
