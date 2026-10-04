// Mock transport tests cover host selection/ABI; actual GPU evidence comes
// from scripts/test-native-browser.mjs against ChromiumRTXCuda and Edge.
import test from 'node:test';
import assert from 'node:assert/strict';
import {GpuRuntime,SupportsNativeCuda,requestPermission} from '../src/runtime/runtime.js';
import {selectNativeBackend} from '../src/runtime/backend.js';
import {NativeRuntime} from '../src/runtime/native-runtime.js';
import {compile,serializableArtifact} from '../src/compiler/compiler.js';

async function navigatorFixture(value,run){const old=Object.getOwnPropertyDescriptor(globalThis,'navigator');Object.defineProperty(globalThis,'navigator',{value,configurable:true});try{return await run();}finally{if(old)Object.defineProperty(globalThis,'navigator',old);else delete globalThis.navigator;}}
function transport({state='granted',available=true,failOpen=false}={}){
 const calls=[];let id=0;
 return {calls,async SupportsNativeCuda(){calls.push('supports');return available;},async queryPermission(){calls.push('query');return state;},async requestPermission(){calls.push('request');state='granted';return state;},async execute(op,payload){calls.push({op,payload:JSON.parse(payload)});if(op==='cuda.open'&&failOpen)throw Error('GPU unavailable');return JSON.stringify({id:++id,session:17});},async close(){calls.push('close');}};
}
test('ordinary browsers select WebGPU without permission or native operations',async()=>navigatorFixture({},async()=>{
 assert.equal(await SupportsNativeCuda(),false);assert.equal((await selectNativeBackend()).status.reason,'unavailable');
 await assert.rejects(requestPermission(),{name:'NotSupportedError'});await assert.rejects(GpuRuntime.create({backend:'native'}),{name:'NotSupportedError'});
}));
test('auto selects native with an existing grant and never asks twice',async()=>{
 const api=transport();await navigatorFixture({cuda:api,userActivation:{isActive:false}},async()=>{
  const runtime=await GpuRuntime.create();assert.equal(runtime.backend,'native-cuda');assert.equal(runtime.describe().native.permission,'granted');assert.ok(!api.calls.includes('request'));
  assert.equal((await selectNativeBackend()).status.reason,'native-initialization-failed');await assert.rejects(GpuRuntime.create({backend:'native'}),{name:'InvalidStateError'});
  await runtime.dispose();const replacement=await GpuRuntime.create({backend:'native'});await replacement.dispose();
 });
});
test('auto prompts only during activation; denied, unavailable and opt-out fall back',async()=>{
 for(const [state,active,allow,reason] of [['prompt',false,true,'permission-required'],['denied',true,true,'permission-denied'],['prompt',true,false,'permission-required']]){
  const api=transport({state});await navigatorFixture({cuda:api,userActivation:{isActive:active}},async()=>{assert.equal((await selectNativeBackend({requestPermission:allow})).status.reason,reason);assert.ok(!api.calls.includes('request'));});
 }
 const api=transport({state:'prompt'});await navigatorFixture({cuda:api,userActivation:{isActive:true}},async()=>{const runtime=await GpuRuntime.create();assert.equal(runtime.backend,'native-cuda');assert.equal(api.calls.filter(x=>x==='request').length,1);await runtime.dispose();});
 const absent=transport({available:false});await navigatorFixture({cuda:absent},async()=>{assert.equal((await selectNativeBackend()).status.reason,'unavailable');assert.deepEqual(absent.calls,['supports']);});
});
test('explicit WebGPU/device/adapter selection does not probe native',async()=>{
 const api=transport();await navigatorFixture({cuda:api},async()=>{
  for(const options of [{backend:'webgpu'},{device:{}},{adapter:{}}])assert.equal((await selectNativeBackend(options)).status.reason,'webgpu-requested');
  assert.deepEqual(api.calls,[]);await assert.rejects(selectNativeBackend({backend:'native',device:{}}),/WebGPU device/);
 });
 await assert.rejects(GpuRuntime.create({backend:'cuda'}),/backend must/);await assert.rejects(GpuRuntime.create({requestPermission:'yes'}),/boolean/);
});
test('strict native errors never turn into silent fallback; auto startup failure is reported',async()=>{
 const api=transport({failOpen:true});await navigatorFixture({cuda:api},async()=>{assert.equal((await selectNativeBackend()).status.reason,'native-initialization-failed');await assert.rejects(GpuRuntime.create({backend:'native'}),/GPU unavailable/);});
 const denied=transport({state:'denied'});await navigatorFixture({cuda:denied},async()=>{await assert.rejects(GpuRuntime.create({backend:'native'}),{name:'NotAllowedError'});assert.ok(!denied.calls.includes('request'));});
});
const source='__global__ void scale(float factor,const float* src,unsigned n,float* dst){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=src[i]*factor;}';
test('dual artifacts retain source and original ordered ABI through serialization',()=>{
 const artifact=serializableArtifact(compile(source,{includeNativeSource:true,workgroupSize:[64,1,1]}));
 assert.equal(artifact.native.source,source);assert.deepEqual(artifact.native.parameters,[{name:'factor',type:'f32'},{name:'src',type:'buffer'},{name:'n',type:'u32'},{name:'dst',type:'buffer'}]);assert.deepEqual(artifact.native.workgroupSize,[64,1,1]);assert.ok(artifact.wgsl);
 assert.equal(serializableArtifact(compile(source)).native,undefined);
 assert.throws(()=>compile(source,{includeNativeSource:'true'}),/boolean/);
 const vector=compile('__global__ void fill(float4* a){a[0].x=1;}',{includeNativeSource:true});
 assert.deepEqual(vector.native.parameters,[{name:'a',type:'buffer'}]);assert.equal(vector.metadata.bindings[0].stride,16);
 assert.throws(()=>compile('__global__ void bad(float3* a){a[0].x=1;}',{includeNativeSource:true}),/incompatible CUDA\/WGSL layouts/);
 assert.throws(()=>compile('__device__ float g[4];__global__ void bad(float* a){a[0]=g[0];}',{includeNativeSource:true}),/synthesized/);
});
test('native artifact execution preserves argument order, scalar snapshots and transactionality',async()=>{
 const api=transport(),runtime=new NativeRuntime(api,7),a=runtime.createBuffer(new Float32Array([1,2])),b=runtime.createBuffer(8);
 const artifact=serializableArtifact(compile(source,{includeNativeSource:true}));const kernel=await runtime.kernel(artifact),inv=kernel.bind({dst:b,src:a},{factor:2,n:2});const batch=runtime.batch().dispatch(inv,[1]);
 inv.setScalars({factor:3});batch.dispatch(inv,[1]);assert.throws(()=>inv.setScalars({factor:4,n:-1}),/Invalid/);assert.equal(inv.scalars.factor,3);
 assert.throws(()=>inv.setScalars({src:5}),/Unknown scalar/);assert.throws(()=>inv.setScalars({factor:1e100}),/Invalid scalar/);batch.submit();await runtime.idle();
 const jobs=api.calls.find(c=>c.op==='cuda.dispatch').payload.jobs;assert.deepEqual(jobs[0].arguments,[{type:'f32',value:2},{buffer:a.id},{type:'u32',value:2},{buffer:b.id}]);assert.equal(jobs[1].arguments[0].value,3);
 assert.equal(await runtime.kernel(artifact),kernel);assert.equal(runtime.stats.pipelineCompiles,1);assert.equal(runtime.stats.pipelineCacheHits,1);assert.equal(runtime.stats.dispatches,2);
 await assert.rejects(runtime.kernel(serializableArtifact(compile(source))),/WGSL-only/);assert.throws(()=>runtime.createTexture2D(),{name:'NotSupportedError'});assert.throws(()=>runtime.batch({timestampWrites:{}}),{name:'NotSupportedError'});assert.throws(()=>runtime.createBuffer(4,{usage:32}),{name:'NotSupportedError'});
 assert.doesNotThrow(()=>runtime.batch().dispatch(inv,[65536,2]));await assert.rejects(runtime.kernel(source,{entry:'scale',workgroupSize:[1025]}),/1024/);
 const pending=runtime.batch().dispatch(inv,[1]);runtime.destroyBuffer(a);assert.throws(()=>pending.submit(),/destroyed/);await runtime.dispose();
});
test('dual artifacts carry default arguments and reject non-buffer scalar bindings',async()=>{
 const api=transport(),runtime=new NativeRuntime(api,1),buffer=runtime.createBuffer(4),artifact=compile('__global__ void fill(float* dst,float value=4){dst[0]=value;}',{includeNativeSource:true});
 const kernel=await runtime.kernel(artifact),inv=kernel.bind({dst:buffer});assert.equal(inv.scalars.value,4);assert.throws(()=>kernel.bind({dst:buffer,value:buffer}),/Unknown buffer/);runtime.batch().dispatch(inv,[1]).submit();await runtime.idle();await runtime.dispose();
});
test('native failure reports once and does not silently migrate live buffers',async()=>{
 const losses=[],api=transport();api.execute=async()=>{throw Error('Permission revoked');};const runtime=new NativeRuntime(api,1,{onError:e=>losses.push(e.message)});runtime.createBuffer(4);
 await assert.rejects(runtime.idle(),/Permission revoked/);await new Promise(r=>setImmediate(r));assert.deepEqual(losses,['Permission revoked']);assert.equal(runtime.backend,'native-cuda');assert.throws(()=>runtime.createBuffer(4),/Permission revoked/);await runtime.dispose();
});
