import test from 'node:test';
import assert from 'node:assert/strict';
import {FLOAT64_WGSL} from '../../src/compiler/float64.js';
import {optimizeLimbMultiply} from './wgsl-limb-multiply.js';
test('limb experiment changes only the known multiplication body and preserves metadata',()=>{
 const input={wgsl:FLOAT64_WGSL,metadata:{entry:'main'}};
 const result=optimizeLimbMultiply(input);
 assert.equal(result.replacements,1);assert.equal(input.wgsl,FLOAT64_WGSL);assert.equal(result.artifact.metadata,input.metadata);
 assert.ok(!result.artifact.wgsl.includes('i < 53u'));assert.ok(!result.artifact.wgsl.includes('i < 49u+extra'));
 assert.equal(result.artifact.wgsl.split('fn cw_d_div')[1],FLOAT64_WGSL.split('fn cw_d_div')[1]);
 assert.equal(optimizeLimbMultiply(result.artifact).replacements,0);
});
test('limb experiment refuses an unknown modified helper body',()=>{
 const input={wgsl:FLOAT64_WGSL.replace('i < 53u','i < 52u')};
 assert.equal(optimizeLimbMultiply(input).artifact,input);
});
