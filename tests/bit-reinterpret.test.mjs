import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('CUDA scalar bit reinterpretation preserves finite IEEE bits and signed zero',()=>{
  const source=`__global__ void bits(const unsigned* input,unsigned* output,int* signedOutput){unsigned i=threadIdx.x;float f=__uint_as_float(input[i]);output[i]=__float_as_uint(f);signedOutput[i]=__float_as_int(__int_as_float((int)input[i]));}`;
  const c=compile(source,{workgroupSize:[8,1,1]});
  assert.match(c.wgsl,/bitcast<f32>/);assert.match(c.wgsl,/bitcast<u32>/);assert.match(c.wgsl,/bitcast<i32>/);
  const input=new Uint32Array([0,0x80000000,0x3f800000,0xbf800000,0x00800000,0x7f7fffff,0x7f800000,0xff800000]);
  const output=new Uint32Array(8),signedOutput=new Int32Array(8);
  executeCPU(c,{input,output,signedOutput},{},[1]);
  assert.deepEqual(output,input);assert.deepEqual(new Uint32Array(signedOutput.buffer),input);
});
for(const [name,arg] of [['__float_as_uint','1u'],['__float_as_int','1'],['__uint_as_float','1.0f'],['__int_as_float','1u']])test(name+' rejects a wrong argument type',()=>{
  assert.throws(()=>compile(`__global__ void bad(float* out){out[0]=(float)${name}(${arg});}`),/requires one/);
});
test('bit reinterpretation rejects incorrect arity',()=>{
  assert.throws(()=>compile('__global__ void bad(float* out){out[0]=__uint_as_float();}'),/requires one/);
});
