import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import {readFileSync} from 'node:fs';

test('scoped enum template entries and forwarded helpers select original shifting variants', () => {
  const source = `enum class Method { NONE, PUSH, XSPH };
    template<Method M> __device__ int selected() {
      if constexpr(M == Method::XSPH) return 42;
      else return 13;
    }
    template<Method M> __global__ void k(int* out) {out[0]=selected<M>();}
    __global__ void named(int* out) {out[0]=selected<Method::PUSH>();}`;
  for(const [entry,value] of [['k<Method::XSPH>',42],['k<Method::PUSH>',13],['named',13]]){
    const artifact=compile(source,{entry,workgroupSize:[1]}),out=new Int32Array(1);
    executeCPU(artifact,{out},{},[1]);assert.equal(out[0],value);
  }
  const chrono=readFileSync(new URL('chrono-shifting.cu',import.meta.url),'utf8');
  for(const method of ['PPST','XSPH','PPST_XSPH','DIFFUSION','DIFFUSION_XSPH'])
    assert.ok(compile(chrono,{entry:`Calc_Shifting_D<ShiftingMethod::${method}>`,defines:{__CUDA_ARCH__:1},workgroupSize:[128]}).wgsl.length>0);
});

test('constexpr selects nested helper template branches before instantiation', () => {
  const source = `
    template<int N> __device__ int missing() { return unknown_call(); }
    template<int N> __device__ int unavailable() { return unknown_call(); }
    template<int N> __device__ int select() {
      if constexpr (N == 2 || N == 3) {
        if constexpr (N == 2) return 17;
        else return 29;
      } else { return missing<N>(); }
    }
    template<int N> __global__ void k(int* out) {
      if constexpr (N > 1 && N < 4) out[0] = select<N>();
      else out[0] = unavailable<N>();
    }`;
  for (const [n, expected] of [[2, 17], [3, 29]]) {
    const artifact = compile(source, {entry: `k<${n}>`, workgroupSize: [1]});
    const out = new Int32Array(1);
    executeCPU(artifact, {out}, {}, [1]);
    assert.equal(out[0], expected);
    assert.doesNotMatch(artifact.wgsl, /missing|unavailable/);
  }
});

test('constexpr short circuits and preserves the selected branch scope', () => {
  const artifact = compile(`__global__ void k(int* out) {
    int a=4;
    if constexpr (false && (1/0)) { out[0]=99; }
    if constexpr (true || (1/0)) { int a=7; out[0]=a; }
    out[1]=a;
  }`, {workgroupSize:[1]});
  const out=new Int32Array(2);
  executeCPU(artifact,{out},{},[1]);
  assert.deepEqual([...out],[7,4]);
});

test('constexpr rejects runtime conditions and invalid constant arithmetic', () => {
  for (const condition of ['threadIdx.x == 0', '1/0', '2147483647+1', '1.5'])
    assert.throws(() => compile(`__global__ void k(int* out){if constexpr (${condition}) out[0]=1;}`), /if constexpr/);
});
