import test from 'node:test';
import assert from 'node:assert/strict';
import {NativeInterop} from '../src/runtime/native-interop.js';

function fixture(failAt=Infinity) {
  let nextId=0;
  const created=[],submissions=[];
  const runtime={device:{limits:{maxTextureDimension2D:8192},lost:new Promise(()=>{})},
    buffers:new Set(),textures:new Set(),assertAlive(){},
    checkResource(r){if(r.runtime!==this||r.destroyed)throw Error('Invalid resource');}};
  const api={async createCanvasSurface(){
    if(++nextId===failAt)throw Error('Allocation failed');
    const surface={id:nextId,available:true,texture:null,buffer:null,
      present(){if(!this.available)throw Error('Surface still presented');this.available=false;},
      destroy(){this.id=0;this.available=false;}};
    created.push(surface);return surface;
  },async dispatchShared(resources,payload){submissions.push({resources,payload:JSON.parse(payload)});return JSON.stringify({resourceCount:resources.length});},close(){}};
  const native=new NativeInterop(runtime,api,1,{canvasPresentation:true,maxBlocksPerLaunch:65536});
  const context={},canvas={width:64,height:64,getContext:()=>context};
  return {native,canvas,created,submissions};
}

test('native canvas pool is bounded and only reuses compositor-released surfaces',async()=>{
  const {native,canvas,created}=fixture(),target=await native.createCanvasTarget(canvas);
  const frames=[target.acquire(),target.acquire(),target.acquire()];
  assert.equal(new Set(frames).size,3);assert.equal(target.acquire(),null);
  for(const frame of frames)target.present(frame);
  assert.equal(target.acquire(),null);
  created[1].available=true;
  assert.equal(target.acquire(),frames[1]);assert.equal(target.acquire(),null);
  target.cancel(frames[1]);assert.equal(target.acquire(),frames[1]);
  canvas.width=128;assert.throws(()=>target.acquire(),/resizing/);
  target.destroy();await native.tail;
  assert.equal(native.stats.sharedBytes,0);assert.ok(created.every(s=>s.id===0));
  assert.throws(()=>target.acquire(),/destroyed/);await native.dispose();
});

test('native canvas surfaces bind to CUDA batches without a GPUTexture',async()=>{
  const {native,canvas,submissions}=fixture(),target=await native.createCanvasTarget(canvas,{buffers:2});
  const image=target.acquire(),kernel={interop:native,id:7,block:[8,8,1],sharedMemoryBytes:0,
    parameters:[{name:'image',type:'surface'}]};
  const result=await native.batch().dispatch({kernel,resources:{image},scalars:{}},[8,8]).submit();
  assert.equal(result.resourceCount,1);assert.equal(image.gpuTexture,undefined);
  assert.deepEqual(submissions[0].payload.jobs[0].arguments,[{surface:image.nativeResource.id}]);
  assert.throws(()=>native.check(image,'buffer'),/shared buffer/);
  target.present(image);target.destroy();await native.dispose();
});

test('failed pool allocation releases earlier surfaces without poisoning the session',async()=>{
  const {native,canvas,created}=fixture(3);
  await assert.rejects(native.createCanvasTarget(canvas),/Allocation failed/);
  assert.ok(created.every(s=>s.id===0));assert.equal(native.resources.size,0);
  assert.equal(native.stats.sharedBytes,0);assert.equal(native.failure,null);
  const target=await native.createCanvasTarget(canvas,{buffers:2});
  assert.ok(target.acquire());target.destroy();await native.dispose();
});


test('explicit idle drains native GPU work after pending submissions',async()=>{
  const {native}=fixture();const order=[];let release;
  native.api.execute=async(command,payload)=>{order.push(command);assert.equal(JSON.parse(payload).$session,1);await new Promise(r=>release=r);return '{}';};
  native.enqueue(async()=>order.push('submit'));
  let complete=false;const waiting=native.idle().then(()=>{complete=true;});
  while(!release)await Promise.resolve();
  assert.deepEqual(order,['submit','cuda.idle']);assert.equal(complete,false);
  release();await waiting;assert.equal(complete,true);await native.dispose();
});
