import test from 'node:test';import assert from 'node:assert/strict';
import {optimizeFloatComparisons} from './wgsl-float-comparisons.js';import {optimizeFloatScaling} from './wgsl-float-scaling.js';
import {specializeIntegerUniforms,guardSpecializedKernel} from './wgsl-specialize.js';
const artifact=expr=>({entryPoint:'main',wgsl:`fn main(){let result=${expr};}`,metadata:{}});
test('narrows only exact promoted-float comparisons',()=>{const r=optimizeFloatComparisons(artifact('cw_d_lt(cw_d_from_f32(x),cw_d_neg(vec2<u32>(0u,1071644672u)))'));assert.equal(r.replacements,1);assert.match(r.artifact.wgsl,/cw_trim_f32_lt\(x, bitcast<f32>\(3204448256u\)\)/);});
test('does not narrow a nonrepresentable double constant',()=>{assert.equal(optimizeFloatComparisons(artifact('cw_d_lt(cw_d_from_f32(x),vec2<u32>(2576980378u,1069128089u))')).replacements,0);});
test('does not narrow actual double arithmetic or double variables',()=>{for(const expr of ['cw_d_lt(cw_d_add(cw_d_from_f32(x),d),cw_d_from_f32(y))','cw_d_eq(d,cw_d_from_f32(y))'])assert.equal(optimizeFloatComparisons(artifact(expr)).replacements,0);});
test('preserves each promoted operand expression once',()=>{const r=optimizeFloatComparisons(artifact('cw_d_eq(cw_d_from_f32(readA()),cw_d_from_f32(readB()))'));assert.equal(r.replacements,1);assert.equal(r.artifact.wgsl.match(/readA\(\)/g).length,1);assert.equal(r.artifact.wgsl.match(/readB\(\)/g).length,1);});
test('scales either operand order by a representable power of two',()=>{for(const expr of ['cw_d_mul(cw_d_from_f32(x),vec2<u32>(0u,1073741824u))','cw_d_mul(vec2<u32>(0u,1073741824u),cw_d_from_f32(x))']){const r=optimizeFloatScaling(artifact(expr));assert.equal(r.replacements,1);assert.match(r.artifact.wgsl,/cw_trim_scale_f32\(x, 1i\)/);}});
test('refuses nonpowers, negatives, excessive shifts, and arbitrary doubles',()=>{for(const expr of ['cw_d_mul(cw_d_from_f32(x),vec2<u32>(0u,1074266112u))','cw_d_mul(cw_d_from_f32(x),vec2<u32>(0u,3221225472u))','cw_d_mul(cw_d_from_f32(x),vec2<u32>(0u,2146435072u))','cw_d_mul(d,vec2<u32>(0u,1073741824u))'])assert.equal(optimizeFloatScaling(artifact(expr)).replacements,0);});
test('numeric passes preserve artifact metadata and input',()=>{const a=artifact('cw_d_mul(cw_d_from_f32(x),vec2<u32>(0u,1073741824u))'),source=a.wgsl,r=optimizeFloatScaling(a);assert.equal(a.wgsl,source);assert.equal(r.artifact.metadata,a.metadata);});
test('reserved helper name collisions are refused',()=>{assert.equal(optimizeFloatScaling(artifact('cw_trim_scale_f32(x,1i)')).replacements,0);assert.equal(optimizeFloatComparisons(artifact('cw_trim_f32_eq(x,y)')).replacements,0);});
test('explicit specialization handles signed integer limits and leaves floats dynamic',()=>{
 const a={entryPoint:'main',wgsl:'struct U { x:i32, y:f32, } @group(0) @binding(0) var<uniform> cw_params:U; fn main(){let a=cw_params.x;let b=cw_params.y;}',metadata:{scalars:[{name:'x',field:'x',type:'i32',origin:'constant'},{name:'y',field:'y',type:'f32',origin:'constant'}]}};
 const r=specializeIntegerUniforms(a,{x:-2147483648,y:3});assert.equal(r.fixed.length,1);assert.match(r.artifact.wgsl,/bitcast<i32>\(2147483648u\)/);assert.match(r.artifact.wgsl,/cw_params.y/);assert.doesNotMatch(r.artifact.wgsl,/cw_params.x/);
});
test('specialization guards updates, defaults, and GPU counters',()=>{
 const kernel=guardSpecializedKernel({bind(buffers,values){return {values:{...values},setScalars(update){Object.assign(this.values,update);return this;}};}},[{name:'mode',value:1,defaultValue:1}]);
 const invocation=kernel.bind({},{});invocation.setScalars({n:7});assert.equal(invocation.values.n,7);assert.throws(()=>invocation.setScalars({mode:0}),/Recompile/);assert.equal(invocation.values.mode,undefined);assert.throws(()=>kernel.bind({},{mode:0}),/Recompile/);assert.throws(()=>kernel.bind({},{mode:1},{scalarBuffers:{mode:{}}}),/GPU counters/);
});
