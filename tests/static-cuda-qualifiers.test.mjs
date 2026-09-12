import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';

test('NVIDIA static qualifiers after CUDA annotations preserve emitted code',()=>{
  const body='float twice(float x){return x*2.0f;}';
  const kernel='void run(float *out){out[threadIdx.x]=twice(float(threadIdx.x));}';
  const options={entry:'run',workgroupSize:[32,1,1]};
  const reference=compile('__device__ '+body+'\n__global__ '+kernel,options).wgsl;
  for(const helper of ['__device__ static','__forceinline__ __device__ static','static __device__']) {
    for(const entry of ['__global__ static','static __global__']) {
      assert.equal(compile(helper+' '+body+'\n'+entry+' '+kernel,options).wgsl,reference);
    }
  }
});
