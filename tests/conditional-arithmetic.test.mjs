import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
const kernel='__global__ void k(){}';
test('NVIDIA launch divisibility guard accepts valid sizes and rejects the error branch',()=>{
 const source='#define NUM_STEPS 2048\n#define THREADBLOCK_SIZE 128\n#if NUM_STEPS % THREADBLOCK_SIZE\n#error Bad constants\n#endif\n'+kernel;
 assert.doesNotThrow(()=>compile(source));
 assert.throws(()=>compile(source.replace('2048','2049')));
});
test('Conditional integer arithmetic keeps precedence and undefined names evaluate to zero',()=>{
 assert.doesNotThrow(()=>compile('#if (3 + 5) / 2 - 4 + UNKNOWN\n#error wrong branch\n#else\n'+kernel+'\n#endif'));
 for(const condition of ['1 / 0','2147483647 + 1','f(1)','1.5 * 2'])assert.throws(()=>compile('#if '+condition+'\n'+kernel+'\n#endif'),/Conditional preprocessing/);
});
