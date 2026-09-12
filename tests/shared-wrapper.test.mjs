import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {kernelSource} from '../src/sandbox/import.js';
const wrapper=fs.readFileSync(new URL('./nbody-shared-memory.cuh',import.meta.url),'utf8'),fixture=fs.readFileSync(new URL('./shared-wrapper.cu',import.meta.url),'utf8');
test('Original NVIDIA wrapper maps to an explicitly sized float4 shared allocation',()=>{
 const c=compile(wrapper+'\n'+fixture,{workgroupSize:[4],sharedMemoryBytes:64}),input=Float32Array.from({length:32},(_,i)=>i),output=new Float32Array(32);
 executeCPU(c,{input,output},{},[2]);assert.equal(c.metadata.workgroupStorageBytes,64);
 for(let i=0;i<8;i++)for(let j=0;j<4;j++)assert.equal(output[i*4+j],input[(Math.floor(i/4)*4+3-i%4)*4+j]);
});
test('Desktop importer preserves whole conversion wrappers without counting methods as kernels',()=>{
 const imported=kernelSource('#include <cuda_runtime.h>\n'+wrapper+'\n'+fixture+'\nint main(){return 0;}');assert.equal(imported.functions,2);assert.equal(imported.source.length,('#include <cuda_runtime.h>\n'+wrapper+'\n'+fixture+'\nint main(){return 0;}').length);
 assert.equal(compile(imported.source,{workgroupSize:[4],sharedMemoryBytes:64}).metadata.workgroupStorageBytes,64);
});
test('Shared wrapper parsing is structural rather than dependent on class or storage names',()=>{
 const renamed=wrapper.replaceAll('SharedMemory','TileView').replaceAll('__smem','backing');
 const c=compile(renamed+'\n'+fixture.replaceAll('SharedMemory','TileView'),{workgroupSize:[4],sharedMemoryBytes:64});assert.equal(c.metadata.dynamicSharedMemoryBytes,64);
});
test('Shared wrappers reject unsupported bodies, mismatched pointer types, missing sizes and overlapping declarations',()=>{
 assert.throws(()=>compile(wrapper+'\n'+fixture,{workgroupSize:[4]}),/sharedMemoryBytes/);
 assert.throws(()=>compile(wrapper.replace('return (T *)__smem;','return (T *)__smem + 1;')+'\n'+fixture),/Expected/);
 for(const body of ['float* p=SharedMemory<float4>();','float* p=SharedMemory<float>();float* q=SharedMemory<float>();'])assert.throws(()=>compile(wrapper+'\n__global__ void k(){'+body+'}',{workgroupSize:[4],sharedMemoryBytes:64}),/matching|Only one dynamic/);
});
