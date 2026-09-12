import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=readFileSync(new URL('native-tiles.cu',import.meta.url),'utf8');
test('Static CUDA tiles explicitly require fixed subgroup size and logical indexing',()=>{const a=compile(source,{workgroupSize:[128]});assert.equal(a.metadata.nativeTiles.size,32);assert.deepEqual(a.metadata.requiredFeatures,['subgroups','subgroup-size-control']);assert.throws(()=>executeCPU(a,{out:new Int32Array(640)},{},[1]),/GPU execution/);for(const workgroupSize of [[16],[48],[32,2]])assert.throws(()=>compile(source,{workgroupSize}),/complete 32-thread/);});
test('Tile memory synchronization rejects partial tile branches',()=>{assert.throws(()=>compile(source.replace('int iterations=0;','if(lane<16){cg::sync(tile);}int iterations=0;'),{workgroupSize:[128]}),/complete tile branch/);});

const phases=readFileSync(new URL('native-tile-phases.cu',import.meta.url),'utf8');
test('Complete tile memory phases compile with workgroup synchronization',()=>{compile(phases,{workgroupSize:[128]});compile(phases.replace('warp==0','warp==2'),{workgroupSize:[128]});});
test('Tile phase loops reject lane bounds, early exits, escaping induction and collectives',()=>{
 for(const [from,to,pattern]of [['row<4','row<threadIdx.x',/literal bounds/],['sum+=tmp;','break;',/early/],['sum+=tmp;','row+=1;',/induction/],['sum+=tmp;','sum+=tile.shfl(sum,0);',/contain collectives/]])assert.throws(()=>compile(phases.replace(from,to),{workgroupSize:[128]}),pattern);
});

test('Vote-loop proof rejects conditional collectives and early exits',()=>{for(const body of ['if(lane<16)iterations+=tile.shfl(lane,0);','if(lane<16)break;','continue;','return;'])assert.throws(()=>compile(source.replace('iterations++;', '{'+body+'}'),{workgroupSize:[128]}),/unconditional collectives|early/);});

test('CUDA popcount preserves all 32 bits in the typed reference',()=>{const a=compile('__global__ void k(unsigned*input,int*out){out[threadIdx.x]=__popc(input[threadIdx.x]);}',{workgroupSize:[4]}),out=new Int32Array(4);executeCPU(a,{input:new Uint32Array([0,0xffffffff,0x80000000,0x55555555]),out},{},[1]);assert.deepEqual([...out],[0,32,1,16]);});
