import test from 'node:test';import assert from 'node:assert/strict';import {compile,serializableArtifact,parse} from '../src/compiler/compiler.js';
const source='__global__ void child(float* out,int i){out[i]=1.f;} __global__ void parent(float* out){child<<<1,32>>>(out,2);}';
const options={entry:'parent',objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:4}};
test('GPU queue metadata retains the original child binding and scalar argument',()=>{
 const a=compile(source,options),q=a.metadata.deviceLaunchQueue.queues[0];assert.equal(a.metadata.deviceLaunchQueue.producerOnly,true);assert.equal(q.byteLength,80);assert.deepEqual(q.block,[32,1,1]);assert.deepEqual(q.buffers,[{name:'out',parent:'out',type:'f32'}]);assert.equal(q.scalars[0].name,'i');
});
test('Device queue bounds and launch interfaces are explicit',()=>{
 for(const maxLaunches of [0,-1,65536,1.5])assert.throws(()=>compile(source,{...options,deviceLaunchQueue:{maxLaunches}}),/maxLaunches/);
 assert.throws(()=>compile(source,{...options,objectHeap:'invocation'}),/persistent/);
 assert.throws(()=>compile(source.replace('>>>(out,2)','>>>(out+1,2)'),options),/named parent buffers/);
 assert.throws(()=>compile(source.replace('<<<1,32>>>','<<<1,32,16,0>>>'),options),/grid and block/);
 assert.throws(()=>compile(source.replace('<<<1,32>>>','<<<1,threadIdx.x>>>'),options),/constant/);
});

test('Scheduled child reads scalar arguments from the GPU queue and retains its original block size',()=>{
 const parent=compile(source,{...options,scheduleDeviceLaunches:true}),child=parent.children[0].artifact;
 assert.equal(serializableArtifact(parent).children[0].artifact.wgsl,child.wgsl);
 assert.deepEqual(child.metadata.scalars,[{name:'cw_launch_slot',type:'u32',offset:0}]);assert.deepEqual(child.metadata.workgroupSize,[32,1,1]);
 assert.match(child.wgsl,/cw_launch_queue_0.words\[cw_params.p_cw_launch_slot\*4u\+3u\]/);
 assert.throws(()=>compile(source,{...options,entry:'child',deviceLaunchConsumer:0,workgroupSize:[64]}),/block size/);
 assert.throws(()=>compile(source,{...options,deviceLaunchConsumer:0}),/leaf child/);
});

test('Templated recursive launches retain configuration and arguments during parsing',()=>{const ast=parse('template<int N> __global__ void tree(int*out){if(threadIdx.x==0)tree<N><<<4,N,4*sizeof(int)>>>(out);}');assert.ok(ast.functions[0]);assert.match(JSON.stringify(ast.functions[0].body),/device-launch/);assert.match(JSON.stringify(ast.functions[0].body),/templateArgument/);});
test('Queued children inherit dynamic shared allocation from CUDA launch',()=>{const s='__global__ void child(int*out){extern __shared__ int s[];s[threadIdx.x]=threadIdx.x;__syncthreads();out[threadIdx.x]=s[31-threadIdx.x];}__global__ void parent(int*out){const int threads=32;const int warps=threads/32;child<<<1,threads,warps*32*sizeof(int)>>>(out);}';const a=compile(s,{...options,scheduleDeviceLaunches:true});assert.equal(a.metadata.deviceLaunchQueue.queues[0].sharedMemoryBytes,128);assert.equal(a.children[0].artifact.metadata.workgroupStorageBytes,128);assert.throws(()=>compile(s,{...options,entry:'child',deviceLaunchConsumer:0,workgroupSize:[32],sharedMemoryBytes:64}),/shared bytes/);for(const n of ['-4','16385','threadIdx.x'])assert.throws(()=>compile(s.replace('warps*32*sizeof(int)',n),options),/shared bytes/);});

test('Scheduled integer-template children retain their selected specialization',()=>{const s='template<int N> __global__ void child(float*out){out[threadIdx.x]=N;}__global__ void parent(float*out){child<8*4><<<1,32>>>(out);}';const a=compile(s,{...options,scheduleDeviceLaunches:true});assert.equal(a.metadata.deviceLaunchQueue.queues[0].childEntry,'child<32>');assert.equal(a.children[0].artifact.metadata.templateArguments.N,32);assert.throws(()=>compile(s.replace('8*4','threadIdx'),options),/constant integers/);});
