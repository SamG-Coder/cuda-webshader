import test from 'node:test';import assert from 'node:assert/strict';import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {kernelSource} from '../src/sandbox/import.js';
const source=`#define SMEM(X, Y) sdata[(Y) * tilew + (X)]
#define ZERO 0
__global__ void k(int* out){int tilew=2;int sdata[4];
#ifndef ABSENT
SMEM(1,0)=7;
#else
unsupported junk
#endif
#ifdef ZERO
SMEM(0,1)=SMEM(1,0)*3;
#endif
#ifdef SMEM
out[0]=2*SMEM(0,1)+1;
#endif
#if 0
#ifdef ABSENT
bad stuff
#else
bad stuff
#endif
#endif
out[1]=SMEM(1,0);}`;
test('Indexed macros retain lvalue assignment and surrounding expression precedence',()=>{const c=compile(source,{workgroupSize:[1]}),out=new Int32Array(2);executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[43,7]);});
test('Desktop extraction retains indexed macros and function-local conditionals',()=>{const s=kernelSource('#include <cuda_runtime.h>\n'+source+'\nint main(){}').source;assert.match(s,/#define SMEM/);assert.doesNotThrow(()=>compile(s));});
test('Definition conditions recognize supplied zero and forwarding macros',()=>{for(const prefix of ['#define FLAG(x) abs(x)\n','']){const c=compile(prefix+'__global__ void k(int* out){\n#ifdef FLAG\nout[0]=9;\n#else\nout[0]=2;\n#endif\n}',{defines:prefix?{}:{FLAG:0},workgroupSize:[1]}),out=new Int32Array(1);executeCPU(c,{out},{},[1]);assert.equal(out[0],9);}});
test('Indexed macro recognition rejects precedence hazards and mismatched delimiters',()=>{for(const body of ['a[x]','a[(x)]+1','a[(x)]*2','a[(x])','a[(x)]++'])assert.throws(()=>compile('#define M(x) '+body+'\n__global__ void k(int* out){int a[3];out[0]=M(1);}'));for(const directive of ['#ifdef X Y','#ifndef','#ifdef (X)'])assert.throws(()=>compile(directive+'\n#endif\n__global__ void k(){}'),/one macro name/);});
