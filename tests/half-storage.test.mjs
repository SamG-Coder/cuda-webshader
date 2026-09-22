import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';

test('Half2 storage uses direct native vec2<f16> loads and stores with four-byte records',()=>{
  const artifact=compile('__global__ void products(const __half2* input,__half2* output){unsigned i=threadIdx.x;output[i]=__hmul2(input[i],__floats2half2_rn(2.0f,-2.0f));}',{workgroupSize:[4,1,1]});
  assert.match(artifact.wgsl,/var<storage, read> b_input: array<vec2<f16>>;/);
  assert.match(artifact.wgsl,/var<storage, read_write> b_output: array<vec2<f16>>;/);
  assert.match(artifact.wgsl,/b_output\[v_i\] = \(b_input\[\w+\] \* vec2<f16>/);
  assert.doesNotMatch(artifact.wgsl,/unpack2x16float|pack2x16float|bitcast/);
  assert.deepEqual(artifact.metadata.requiredFeatures,['shader-f16']);
  assert.deepEqual(artifact.metadata.bindings.map(({name,elementType,stride,readOnly})=>({name,elementType,stride,readOnly})),[
    {name:'input',elementType:'vec2<f16>',stride:4,readOnly:true},
    {name:'output',elementType:'vec2<f16>',stride:4,readOnly:false},
  ]);
  // Upload ABI is raw binary16, independent of the JavaScript view's element type.
  const input=new Uint32Array([0xbc003c00,0x00018000,0x35553c01,0x7bff0400]),output=new Uint32Array(4);
  executeCPU(artifact,{input,output},{},[1]);
  assert.deepEqual([...output],[0x40004000,0x80028000,0xb9554001,0xfc000800]);
});

test('Scalar half storage round trips every finite binary16 code, both zeros and infinities',()=>{
  const artifact=compile('__global__ void copy(const __half* input,__half* output,float* decoded,unsigned count){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){output[i]=input[i];decoded[i]=__half2float(input[i]);}}',{workgroupSize:[64,1,1]});
  assert.match(artifact.wgsl,/var<storage, read> b_input: array<f16>;/);
  assert.match(artifact.wgsl,/var<storage, read_write> b_output: array<f16>;/);
  assert.equal(artifact.metadata.bindings[0].stride,2);
  assert.equal(artifact.metadata.bindings[1].stride,2);
  const input=Uint16Array.from({length:65536},(_,i)=>i),output=new Uint16Array(input.length),decoded=new Float32Array(input.length);
  executeCPU(artifact,{input,output,decoded},{count:input.length},[1024]);
  for(let code=0;code<input.length;code++){
    if((code&0x7fff)>0x7c00){assert(Number.isNaN(decoded[code]));assert((output[code]&0x7fff)>0x7c00);}
    else assert.equal(output[code],code,'binary16 code '+code.toString(16));
  }
  assert(Object.is(decoded[0x8000],-0));assert.equal(decoded[1],2**-24);
  assert.equal(decoded[0x7bff],65504);assert.equal(decoded[0xfc00],-Infinity);
});

test('Half storage oracle writes rounded encodings and respects byte views and helper offsets',()=>{
  const artifact=compile('__device__ void store(__half* output,float value){output[0]=__float2half_rn(value);} __global__ void convert(const float* input,__half* output){unsigned i=threadIdx.x;store(output+i,input[i]);}',{workgroupSize:[8,1,1]});
  const input=new Float32Array([1.00048828125,1.00146484375,65504,65520,2**-25,3*2**-25,-2,-0]);
  const bytes=new Uint8Array(18);bytes.fill(0xa5);const output=bytes.subarray(1,17);
  executeCPU(artifact,{input,output},{},[1]);
  const view=new DataView(output.buffer,output.byteOffset,output.byteLength);
  assert.deepEqual(Array.from({length:8},(_,i)=>view.getUint16(i*2,true)),[0x3c00,0x3c02,0x7bff,0x7c00,0,2,0xc000,0x8000]);
  assert.equal(bytes[0],0xa5);assert.equal(bytes[17],0xa5);
});

test('Half pointer-only kernels declare the feature and reject const writes and partial records',()=>{
  const artifact=compile('__global__ void copy(const __half2* input,__half2* output){output[0]=input[0];}');
  assert.deepEqual(artifact.metadata.requiredFeatures,['shader-f16']);
  assert.match(artifact.wgsl,/enable f16;/);
  assert.throws(()=>compile('__global__ void invalid(const __half2* output){output[0]=__floats2half2_rn(0.0f,0.0f);}'),/const/);
  assert.throws(()=>executeCPU(artifact,{input:new Uint8Array(6),output:new Uint32Array(1)},{},[1]),/complete binary16 records/);
  const scalar=compile('__global__ void copy(const __half* input,__half* output){output[0]=input[0];}');
  assert.throws(()=>executeCPU(scalar,{input:new Uint8Array(1),output:new Uint16Array(1)},{},[1]),/complete binary16 records/);
});
