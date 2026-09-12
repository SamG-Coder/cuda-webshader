import {readFileSync} from 'node:fs';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
import test from 'node:test';import assert from 'node:assert/strict';import {compile,serializableArtifact,parse} from '../src/compiler/compiler.js';
const source='__global__ void child(float* out,int i){out[i]=1.f;} __global__ void parent(float* out){child<<<1,32>>>(out,2);}';
const options={entry:'parent',objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:4}};
test('GPU queue metadata retains the original child binding and scalar argument',()=>{
 const a=compile(source,options),q=a.metadata.deviceLaunchQueue.queues[0];assert.equal(a.metadata.deviceLaunchQueue.producerOnly,true);assert.equal(q.byteLength,96);assert.deepEqual(q.block,[32,1,1]);assert.deepEqual(q.buffers,[{name:'out',parent:'out',type:'f32',argument:0,offsetWord:4}]);assert.equal(q.scalars[0].name,'i');
});
test('Device queue bounds and launch interfaces are explicit',()=>{
 for(const maxLaunches of [0,-1,65536,1.5])assert.throws(()=>compile(source,{...options,deviceLaunchQueue:{maxLaunches}}),/maxLaunches/);
 assert.throws(()=>compile(source,{...options,objectHeap:'invocation'}),/persistent/);
 assert.equal(compile(source.replace('>>>(out,2)','>>>(out+1,2)'),options).metadata.deviceLaunchQueue.queues[0].buffers[0].offsetWord,4);
 assert.throws(()=>compile(source.replace('<<<1,32>>>','<<<1,32,16,0>>>'),options),/grid and block/);
 assert.throws(()=>compile(source.replace('<<<1,32>>>','<<<1,threadIdx.x>>>'),options),/constant/);
});

test('Scheduled child reads scalar arguments from the GPU queue and retains its original block size',()=>{
 const parent=compile(source,{...options,scheduleDeviceLaunches:true}),child=parent.children[0].artifact;
 assert.equal(serializableArtifact(parent).children[0].artifact.wgsl,child.wgsl);
 assert.deepEqual(child.metadata.scalars,[{name:'cw_launch_slot',type:'u32',offset:0}]);assert.deepEqual(child.metadata.workgroupSize,[32,1,1]);
 assert.match(child.wgsl,/cw_launch_queue_0.words\[cw_params.p_cw_launch_slot\*5u\+3u\]/);
 assert.throws(()=>compile(source,{...options,entry:'child',deviceLaunchConsumer:0,workgroupSize:[64]}),/block size/);
 assert.throws(()=>compile(source,{...options,deviceLaunchConsumer:0}),/leaf child/);
});

test('Templated recursive launches retain configuration and arguments during parsing',()=>{const ast=parse('template<int N> __global__ void tree(int*out){if(threadIdx.x==0)tree<N><<<4,N,4*sizeof(int)>>>(out);}');assert.ok(ast.functions[0]);assert.match(JSON.stringify(ast.functions[0].body),/device-launch/);assert.match(JSON.stringify(ast.functions[0].body),/templateArgument/);});
test('Queued children inherit dynamic shared allocation from CUDA launch',()=>{const s='__global__ void child(int*out){extern __shared__ int s[];s[threadIdx.x]=threadIdx.x;__syncthreads();out[threadIdx.x]=s[31-threadIdx.x];}__global__ void parent(int*out){const int threads=32;const int warps=threads/32;child<<<1,threads,warps*32*sizeof(int)>>>(out);}';const a=compile(s,{...options,scheduleDeviceLaunches:true});assert.equal(a.metadata.deviceLaunchQueue.queues[0].sharedMemoryBytes,128);assert.equal(a.children[0].artifact.metadata.workgroupStorageBytes,128);assert.throws(()=>compile(s,{...options,entry:'child',deviceLaunchConsumer:0,workgroupSize:[32],sharedMemoryBytes:64}),/shared bytes/);for(const n of ['-4','16385','threadIdx.x'])assert.throws(()=>compile(s.replace('warps*32*sizeof(int)',n),options),/shared bytes/);});

test('Scheduled integer-template children retain their selected specialization',()=>{const s='template<int N> __global__ void child(float*out){out[threadIdx.x]=N;}__global__ void parent(float*out){child<8*4><<<1,32>>>(out);}';const a=compile(s,{...options,scheduleDeviceLaunches:true});assert.equal(a.metadata.deviceLaunchQueue.queues[0].childEntry,'child<32>');assert.equal(a.children[0].artifact.metadata.templateArguments.N,32);assert.throws(()=>compile(s.replace('8*4','threadIdx'),options),/constant integers/);});

test('Record kernel arguments preserve fields and queued constructor snapshots',()=>{const source=readFileSync(new URL('queued-record.cu',import.meta.url),'utf8'),a=compile(source,{...options,entry:'record_parent',scheduleDeviceLaunches:true});assert.equal(a.metadata.scalars.length,5);assert.equal(a.metadata.deviceLaunchQueue.queues[0].scalars.length,6);assert.equal(a.children[0].artifact.metadata.scalars.length,1);const leaf=compile(source,{entry:'record_child'}),out=new Int32Array(5);executeCPU(leaf,{out},{'params.point_selector':1,'params.num_nodes_at_this_level':4,'params.depth':3,'params.max_depth':8,'params.min_points_per_node':16,index:0},[1]);assert.deepEqual([...out],[1,4,3,8,16]);});
test('Nested record launch fields retain paths and reject pointer-bearing layouts',()=>{const source='struct Inner { int x; bool on; };struct Outer { Inner inner; float y; };__global__ void k(float*out,Outer p){out[0]=p.inner.on?p.inner.x+p.y:0.0f;}';const a=compile(source),out=new Float32Array(1);executeCPU(a,{out},{'p.inner.x':3,'p.inner.on':1,'p.y':.5},[1]);assert.deepEqual([...out],[3.5]);assert.throws(()=>compile('struct R{float v[2];};__global__ void k(R p){}'),/scalar-only/);});

test('Record kernel values are independent in every invocation',()=>{const a=compile('struct R{int n;};__global__ void k(int*out,R p){p.n=p.n+threadIdx.x;out[threadIdx.x]=p.n;}',{workgroupSize:[4]}),out=new Int32Array(4);executeCPU(a,{out},{'p.n':10},[1]);assert.deepEqual([...out],[10,11,12,13]);assert.throws(()=>executeCPU(a,{out},{'p.n':2147483648},[1]),/Invalid record/);});

test('Queued buffer offsets retain allocation identity through aliases',()=>{const s=readFileSync(new URL('queued-offset.cu',import.meta.url),'utf8'),a=compile(s,{...options,entry:'offset_parent',scheduleDeviceLaunches:true});assert.equal(a.metadata.deviceLaunchQueue.queues[0].buffers[0].parent,'out');assert.match(a.children[0].artifact.wgsl,/var cw_pointer_out: i32 = bitcast<i32>/);assert.match(a.wgsl,/arrayLength/);assert.throws(()=>compile('__global__ void child(int*out){}__global__ void parent(const int*out){child<<<1,1>>>(out+1);}',options),/discard const/);assert.throws(()=>compile('__global__ void child(int*out){}__global__ void parent(int*out){int local[4];child<<<1,1>>>(&local[1]);}',options),/named parent buffers/);});

test('Imported scalar buffers retain child offsets',()=>{const s='class Holder{int*p;public:__device__ Holder(int*x):p(x){}};__global__ void child(int*out){out[0]=7;}__global__ void parent(int*out){child<<<1,1>>>(out+2);}';const a=compile(s,{...options,scheduleDeviceLaunches:true});assert.match(a.children[0].artifact.wgsl,/var cw_pointer_out: i32 = bitcast<i32>/);assert.match(a.children[0].artifact.wgsl,/cw_import_1\[\(cw_pointer_out/);});

test('CUDA warpSize is the logical 32-lane width and local names retain scope',()=>{for(const [declaration,width] of [['',32],['const int warpSize=16;',16]]){const a=compile('__global__ void k(int*out){'+declaration+'out[threadIdx.x]=threadIdx.x/warpSize;}',{workgroupSize:[64]}),out=new Int32Array(64);executeCPU(a,{out},{},[1]);for(let i=0;i<64;i++)assert.equal(out[i],Math.floor(i/width));}});
