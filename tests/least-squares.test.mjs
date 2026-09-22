import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {NORMAL_EQUATIONS_SOURCE, solveNormalEquations} from '../src/runtime/least-squares.js';

test('normal-equation CUDA compiles with bounded bindings', () => {
  const artifact = compile(NORMAL_EQUATIONS_SOURCE, {entry:'normal_equations',workgroupSize:[64,1,1]});
  assert.equal(artifact.metadata.bindings.length, 2);
});
test('solves several outputs with partial pivoting', () => {
  // x = -2..2, targets 2x+3 and -x+4; features [x,1].
  const solution = solveNormalEquations([10,0,20,-10,0,5,15,20],2,2,0);
  assert.deepEqual(Array.from(solution),[2,3,-1,4]);
  assert.deepEqual(Array.from(solveNormalEquations([0,1,2,1,1,5],2,1,0)),[3,2]);
});
test('ridge regularizes duplicate features; invalid and empty systems fail', () => {
  const solution = solveNormalEquations([2,2,4,2,2,4],2,1);
  assert.ok(solution.every(v=>Math.abs(v-1)<1e-5));
  assert.throws(()=>solveNormalEquations([2,2,4,2,2,4],2,1,0),/Singular/);
  assert.throws(()=>solveNormalEquations([0,0],1,1),/no nonzero/);
  assert.throws(()=>solveNormalEquations([NaN,1],1,1),/Invalid/);
});
