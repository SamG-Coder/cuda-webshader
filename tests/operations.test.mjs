import test from 'node:test';import assert from 'node:assert/strict';
import {prepareSaxpy,prepareMatmul,prepareReduction} from '../src/runtime/operations.js';
function mock() {
  const allocations = [], launches = [], destroyed = [];
  const runtime = {
    device: {limits: {maxComputeWorkgroupsPerDimension: 65535}},
    checkResource(r) { if (!r || r.destroyed) throw new Error('Invalid resource'); },
    createBuffer(bytes) {
      const r = {size: Math.max(4, bytes), byteLength: bytes};
      allocations.push(r); return r;
    },
    destroyBuffer(r) { r.destroyed = true; destroyed.push(r); },
    async kernel(source, options) {
      return {
        bind(buffers, values) {
          return {buffers, values, options, setScalars(v) { Object.assign(this.values, v); }};
        }
      };
    }
  };
  const batch = {dispatch(inv, groups) { launches.push({inv, groups}); return this; }};
  return {runtime, batch, allocations, launches, destroyed};
}
const sources={saxpy:'scalar',saxpy_vec4:'vec4',matmul_tiled:'tiled',matmul_register:'register',matmul_naive:'naive',reduce_sum:'sum'};
test('A measured vec4 winner falls back to scalar for tail lengths',async()=>{const m=mock(),x=m.runtime.createBuffer(1031*4),y=m.runtime.createBuffer(1031*4),plan=await prepareSaxpy(m.runtime,sources,{x,y,n:1031,choice:{id:'saxpy_vec4',workgroupSize:[64,1,1]}});assert.equal(plan.id,'saxpy');assert.deepEqual(plan.groups,[17,1,1]);plan.encode(m.batch,2);assert.equal(plan.invocation.values.a,2);});
test('Aligned SAXPY retains measured vec4 workgroup choice',async()=>{const m=mock(),x=m.runtime.createBuffer(4096),y=m.runtime.createBuffer(4096),plan=await prepareSaxpy(m.runtime,sources,{x,y,n:1024,choice:{id:'saxpy_vec4',workgroupSize:[256,1,1]}});assert.equal(plan.id,'saxpy_vec4');assert.equal(plan.invocation.values.n4,256);assert.deepEqual(plan.groups,[1,1,1]);});
test('Plans reject insufficient source buffers before creating a pipeline',async()=>{const m=mock(),x=m.runtime.createBuffer(4),y=m.runtime.createBuffer(4);await assert.rejects(prepareSaxpy(m.runtime,sources,{x,y,n:2}),/does not contain/);});
test('Register matrix variant launches in 16x16 output tiles',async()=>{const m=mock(),A=m.runtime.createBuffer(17*23*4),B=m.runtime.createBuffer(23*19*4),C=m.runtime.createBuffer(17*19*4),p=await prepareMatmul(m.runtime,sources,{A,B,C,M:17,N:19,K:23,choice:{id:'matmul_register'}});assert.deepEqual(p.groups,[2,2,1]);assert.deepEqual(p.invocation.options.workgroupSize,[8,8,1]);});
test('Hierarchical reduction preallocates reusable scratch and chains GPU outputs',async()=>{const m=mock(),input=m.runtime.createBuffer(4099*4),p=await prepareReduction(m.runtime,sources,{input,n:4099});assert.equal(p.levels,2);const count=m.allocations.length;p.encode(m.batch);p.encode(m.batch);assert.equal(m.allocations.length,count);assert.equal(m.launches[0].inv.buffers.output,m.launches[1].inv.buffers.input);assert.equal(p.output,m.launches[1].inv.buffers.output);p.dispose();p.dispose();assert.equal(m.destroyed.length,2);assert.throws(()=>p.encode(m.batch),/disposed/);assert.equal(input.destroyed,undefined);});
for(const n of [0,1])test(`Reduction n=${n} still produces a distinct scalar output`,async()=>{const m=mock(),input=m.runtime.createBuffer(n*4),p=await prepareReduction(m.runtime,sources,{input,n});assert.equal(p.levels,1);assert.notEqual(p.output,input);p.encode(m.batch);assert.deepEqual(m.launches[0].groups,[1,1,1]);assert.equal(m.launches[0].inv.values.n,n);p.dispose();});
