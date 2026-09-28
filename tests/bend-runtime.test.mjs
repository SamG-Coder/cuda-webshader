import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {compileBend,lowerBend,checkBend} from '../src/bend/compiler.js';
import {examples} from '../src/bend/examples.js';
import {bendCases} from './bend-cases.js';
import {packBendInputs} from '../src/bend/runtime.js';
const runtime=await readFile(new URL('../src/bend/runtime.cu',import.meta.url),'utf8');

test('Bend examples pass the upstream checker and compile through CUDA to WGSL',()=>{
  for(const sample of bendCases) {
    const result=compileBend(sample.source,runtime,{entry:sample.entry});
    assert.match(result.cuda,/__global__ void bend_run/);
    assert.match(result.artifact.wgsl,/@compute/);
    assert.equal(result.verification.formalVerdict,false);
  }
});
test('Input packing preserves u32/f32 bits and rejects out-of-range host data',()=>{
  const p=lowerBend(examples.closure.source);
  const words=packBendInputs(p,[[1.5,-0]]);
  assert.deepEqual([...words],[0x3fc00000,0x80000000]);
  assert.throws(()=>packBendInputs(p,[[Infinity,0]]),/finite/);
  assert.throws(()=>packBendInputs(p,[[1e100,0]]),/F32 range/);
  assert.throws(()=>packBendInputs(p,[[1]]),/needs 2/);
  const u=lowerBend(examples.proof.source,{entry:'identity'});
  for(const x of [-1,1.5,4294967296])assert.throws(()=>packBendInputs(u,[[x]]),/unsigned 32/);
});
test('Bend rejects false laws, affine misuse, unsafe definitions, holes and foreign imports',()=>{
  assert.throws(()=>checkBend(examples.proof.source.replace('identity(x) == x','identity(x) == 0')),/Error/);
  assert.throws(()=>checkBend('import Base\ndef main(x: U32) -> U32:\n  (x + x : U32)\n'),/Error/);
  assert.throws(()=>checkBend('import Base\n@unsafe def main() -> U32:\n  1\n'),/Unsafe/);
  assert.throws(()=>checkBend('import Base\nlaw missing: U32\n'),/unfilled/);
  assert.throws(()=>checkBend('import ./secret as Secret\n'),/import Base only/);
});
test('Unsupported entry types fail explicitly',()=>{
  assert.throws(()=>lowerBend('import Base\ndef main() -> String:\n  "hello"\n'),/Entry arguments and result/);
});
