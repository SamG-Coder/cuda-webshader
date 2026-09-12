import test from 'node:test';import assert from 'node:assert/strict';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {packScalars} from '../src/runtime/runtime.js';
const source='__global__ void k(float* out,unsigned int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;__shared__ float tile[4];tile[threadIdx.x]=(float)i;__syncthreads();out[i]=tile[3u-threadIdx.x];}',options={workgroupSize:[4],fullWorkgroups:['n']};
test('Full-workgroup contract preserves block-local exchange and excess whole-block returns',()=>{
 const c=compile(source,options),out=new Float32Array(12).fill(-1);executeCPU(c,{out},{n:8},[3]);assert.deepEqual([...out],[3,2,1,0,7,6,5,4,-1,-1,-1,-1]);assert.match(c.wgsl,/cw_block.x >=/);assert.deepEqual(serializableArtifact(c).metadata.scalarConstraints,[{name:'n',multipleOf:4,minimum:0}]);
});
test('Full-workgroup runtime validation rejects partial blocks transactionally',()=>{
 const c=compile(source,options),data=new ArrayBuffer(c.metadata.uniformSize);packScalars(c.metadata,{n:8},data);const before=new Uint8Array(data.slice(0));
 for(const n of [1,3,7,9,-4,NaN]){assert.throws(()=>packScalars(c.metadata,{n},data));assert.deepEqual(new Uint8Array(data),before);assert.throws(()=>executeCPU(c,{out:new Float32Array(12)},{n},[3]));}
});
test('Zero whole-block count leaves buffers unchanged',()=>{
 const c=compile(source,options),out=new Float32Array(4).fill(7);packScalars(c.metadata,{n:0});executeCPU(c,{out},{n:0},[1]);assert.deepEqual([...out],[7,7,7,7]);
});
test('Whole-block lowering is opt-in and rejects unrecognized guard shapes',()=>{
 assert.equal(compile(source,{workgroupSize:[4]}).metadata.scalarConstraints,undefined);
 for(const modified of [source.replace('i>=n','i>n'),source.replace('threadIdx.x','threadIdx.x+1u'),source.replace('if(i>=n)return;','if(i>=n){out[0]=0.0f;return;}'),source.replace('unsigned int n','int n')])assert.throws(()=>compile(modified,options),/Full-workgroup|supported linear/);
 assert.throws(()=>compile(source,{...options,workgroupSize:[2,2,1]}),/rectangular guard/);
});
test('Whole-block lowering rejects bound mutation, shadowing and reference escape',()=>{
 for(const prefix of ['n=4u;','n++;','{unsigned int n=4u;}'])assert.throws(()=>compile(source.replace('int i=',prefix+'int i='),options),/unchanged/);
 assert.throws(()=>compile('__device__ void f(unsigned int& n){n=4u;} '+source.replace('int i=','f(n);int i='),options),/unchanged/);
 for(const fullWorkgroups of [true,[],['n','n'],['missing']])assert.throws(()=>compile(source,{...options,fullWorkgroups}),/fullWorkgroups|Full-workgroup/);
});
