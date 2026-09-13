import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';
test('Runtime integer powers preserve operand evaluation and signed zero',()=>{
 const a=compile('__device__ float next(float& v){float old=v;v+=1.f;return old;}__global__ void k(float* out){float a=2.f;float b=3.f;out[0]=powf(next(a),next(b));out[1]=a;out[2]=b;out[3]=powf(-0.f,3.f);}',{workgroupSize:[1]}),out=new Float32Array(4);executeCPU(a,{out},{},[1]);assert.deepEqual([...out],[8,3,4,-0]);assert.match(a.wgsl,/cw_pow_f32/);
});
test('Captured integer powers agree with an independent scalar oracle',()=>{
 const raw=readFileSync(new URL('../reports/integer-powers-input.bin',import.meta.url)),input=new Float32Array(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)),n=input.length/2,output=new Float32Array(n),a=compile(readFileSync(new URL('integer-powers.cu',import.meta.url),'utf8'),{workgroupSize:[64]});executeCPU(a,{input,output},{n},[Math.ceil(n/64)]);
 for(let i=0;i<n;i++){const e=Math.fround(Math.pow(input[i*2],input[i*2+1]));assert.ok(Object.is(output[i],e)||Number.isNaN(output[i])&&Number.isNaN(e));}
});
