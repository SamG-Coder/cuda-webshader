import test from 'node:test';import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';import {executeCPU} from '../src/compiler/cpu-oracle.js';import {decodeDiagnostics} from '../src/compiler/diagnostics.js';
test('Integer printf capture evaluates arguments once and reports bounded overflow',()=>{
 const a=compile('__global__ void k(int* out){int i=-2;printf("v=%d %%\\n",i++);printf("v=%d %%\\n",i++);printf("u=%u\\n",4294967295u);out[0]=i;}',{workgroupSize:[1],diagnosticCapacity:2});
 const capture=new Uint32Array(a.metadata.bindings.find(b=>b.name===a.metadata.diagnostics.buffer).count),out=new Int32Array(1);
 executeCPU(a,{out,[a.metadata.diagnostics.buffer]:capture},{},[1]);
 assert.equal(out[0],0);assert.deepEqual(decodeDiagnostics(a.metadata,capture),{attempted:3,dropped:1,messages:['v=-2 %\n','v=-1 %\n']});
 assert.throws(()=>decodeDiagnostics(a.metadata,new Uint32Array(1)),/Incomplete/);
});
test('Unsupported printf forms and invalid capture capacities fail explicitly',()=>{
 for(const s of ['__global__ void k(){printf("%f",1.f);}','__global__ void k(){printf("%u",1.f);}','__global__ void k(){printf("%u");}','__global__ void k(){int n=printf("hello");}','__global__ void k(){printf("%s",1);}','__global__ void k(){int cw_printf_reserved=0;printf("hello");}'])assert.throws(()=>compile(s));
 for(const diagnosticCapacity of [0,1.5,65537])assert.throws(()=>compile('__global__ void k(){printf("hello");}',{diagnosticCapacity}));
});
test('Float finite classification distinguishes finite extrema, infinity and NaN',()=>{
 const a=compile('__global__ void k(const float* input,unsigned int* out){unsigned int i=threadIdx.x;out[i]=isfinite(input[i]);}',{workgroupSize:[6]});
 const input=new Float32Array([0,-0,3.4028234663852886e38,Infinity,-Infinity,NaN]),out=new Uint32Array(6);executeCPU(a,{input,out},{},[1]);assert.deepEqual([...out],[1,1,1,0,0,0]);
 assert.throws(()=>compile('__global__ void k(){bool b=isfinite(1.0);}'),/float argument/);
});
