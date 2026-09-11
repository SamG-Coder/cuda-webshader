import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {kernelSource} from '../src/sandbox/import.js';
const prefix='namespace blocks = cooperative_groups;\n',wrap=body=>prefix+'__global__ void k(float* out) {'+body+'}';
for(const sync of ['blocks::sync(team);','team.sync();'])test('Scoped block synchronization: '+sync,()=>{
 const artifact=compile(wrap('blocks::thread_block team=blocks::this_thread_block(); __shared__ float tile[4]; tile[threadIdx.x]=(float)threadIdx.x;'+sync+'out[threadIdx.x]=tile[3u-threadIdx.x];'),{workgroupSize:[4,1,1]});
 assert.match(artifact.wgsl,/workgroupBarrier\(\)/);assert.doesNotMatch(artifact.wgsl,/v_team|thread_block|blocks::/);const out=new Float32Array(4);executeCPU(artifact,{out},{},[1]);assert.deepEqual([...out],[3,2,1,0]);
});
test('Block sync preserves storage visibility when a buffer is read and written',()=>{const a=compile(wrap('blocks::thread_block team=blocks::this_thread_block();out[threadIdx.x]=1.0f;blocks::sync(team);out[threadIdx.x]=out[3u-threadIdx.x];'),{workgroupSize:[4,1,1]});assert.match(a.wgsl,/workgroupBarrier\(\);\n\s*storageBarrier\(\);/);});
test('Divergent block sync remains visible to the CPU barrier oracle',()=>{const a=compile(wrap('blocks::thread_block team=blocks::this_thread_block();if(threadIdx.x==0u)blocks::sync(team);'),{workgroupSize:[4,1,1]});assert.throws(()=>executeCPU(a,{out:new Float32Array(4)},{},[1]),/Nonuniform/);});
for(const [name,body] of [
 ['missing handle','blocks::sync(team);'],
 ['scalar handle','int team=0; blocks::sync(team);'],
 ['scalar member sync','int team=0; team.sync();'],
 ['out of scope','{blocks::thread_block team=blocks::this_thread_block();}blocks::sync(team);'],
 ['shadowed handle','blocks::thread_block team=blocks::this_thread_block();{int team=0;blocks::sync(team);}'],
 ['unknown namespace','fake::thread_block team=fake::this_thread_block();'],
 ['grid group','blocks::grid_group team=blocks::this_grid();'],
 ['factory arguments','blocks::thread_block team=blocks::this_thread_block(1);'],
 ['wrong factory','blocks::thread_block team=blocks::this_grid();'],
 ['extra sync arguments','blocks::thread_block team=blocks::this_thread_block();blocks::sync(team,1);'],
 ['handle arithmetic','blocks::thread_block team=blocks::this_thread_block();out[0]=team+1;'],
 ['handle reassignment','blocks::thread_block team=blocks::this_thread_block();team=1;'],
])test('Reject unsupported cooperative-group use: '+name,()=>assert.throws(()=>compile(wrap(body))));
test('Group handles cannot be smuggled into helper functions',()=>assert.throws(()=>compile(prefix+'__device__ void helper(){blocks::thread_block team=blocks::this_thread_block();blocks::sync(team);}\n__global__ void k() {helper();}'),/only inside a kernel/));
test('Desktop import retains an explicit namespace alias and original line positions',()=>{const source='#include <cooperative_groups.h>\n'+wrap('blocks::thread_block team=blocks::this_thread_block();blocks::sync(team);')+'\nint main(){return 0;}';const imported=kernelSource(source);assert.equal(imported.source.split('\n')[1],source.split('\n')[1]);assert.match(compile(imported.source).wgsl,/workgroupBarrier/);});
for(const [file,entry] of [['18.cu','transposeCoalesced'],['19.cu','transposeNoBankConflicts']])test('NVIDIA '+entry+' transposes a rectangular matrix',async()=>{const source=await readFile(new URL('../showcases/nvidia/kernels/'+file,import.meta.url),'utf8'),a=compile(source,{entry,workgroupSize:[32,16,1]}),width=64,height=32,idata=Float32Array.from({length:width*height},(_,i)=>i/8-100),odata=new Float32Array(width*height);executeCPU(a,{idata,odata},{width,height},[2,1,1]);for(let y=0;y<height;y++)for(let x=0;x<width;x++)assert.equal(odata[x*height+y],idata[y*width+x]);assert.equal(a.metadata.workgroupStorageBytes,entry==='transposeCoalesced'?4096:4224);});
