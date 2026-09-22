import test from 'node:test';
import assert from 'node:assert/strict';
import {compile,typeStride} from '../src/compiler/compiler.js';
import {executeCPU,roundHalf} from '../src/compiler/cpu-oracle.js';

test('Native half and half2 emit feature metadata and compact shared storage',()=>{
  const code=`__global__ void h(const float* input,float* output){unsigned i=threadIdx.x; __shared__ __half2 tile[8];tile[i]=__floats2half2_rn(input[i],input[i]);__syncthreads();float2 v=__half22float2(__hmul2(tile[i],tile[i]));output[i]=v.x;}`;
  const a=compile(code,{workgroupSize:[8,1,1]});
  assert.match(a.wgsl,/enable f16;/);assert.match(a.wgsl,/array<vec2<f16>, 8>/);
  assert.deepEqual(a.metadata.requiredFeatures,['shader-f16']);assert.equal(a.metadata.workgroupStorageBytes,32);assert.equal(typeStride('f16'),2);
  const input=new Float32Array([0,-0,1,-2,1.00048828125,0.25,2**-7,9]),output=new Float32Array(8);
  executeCPU(a,{input,output},{},[1]);assert.deepEqual(output,Float32Array.from(input,x=>roundHalf(roundHalf(x)**2)));
});
test('Half scalar conversions and arithmetic honor binary16 ties and limits in the oracle',()=>{
  const a=compile('__global__ void h(const float* input,float* output){unsigned i=threadIdx.x;__half v=__float2half_rn(input[i]);output[i]=__half2float(__hadd(v,__float2half_rn(0.0f)));}',{workgroupSize:[8,1,1]});
  const input=new Float32Array([1.00048828125,1.00146484375,65504,65520,2**-25,3*2**-25,-2,0]),output=new Float32Array(8);
  executeCPU(a,{input,output},{},[1]);assert.deepEqual([...output],[1,1.001953125,65504,Infinity,0,2**-23,-2,0]);
});
test('Half ABI and wrong intrinsic arguments are rejected explicitly',()=>{
  for(const params of ['__half value','__half2 value'])assert.throws(()=>compile(`__global__ void h(${params}){}`),/Half kernel parameters/);
  assert.throws(()=>compile('__global__ void h(float* out){out[0]=__half2float(1.0f);}'),/requires one f16/);
  assert.throws(()=>compile('__global__ void h(float* out){__half2 x=__floats2half2_rn(1,2);}'),/requires two float/);
  const ordinary=compile('__global__ void h(float* out){unsigned half=2;out[0]=(float)(half-1);}');assert.doesNotMatch(ordinary.wgsl,/enable f16/);
});
