export async function checkBezierParent(runtime){
 const source=await(await fetch('/tests/bezier-cdp-device.cuh')).text(),controls=new Float32Array(await(await fetch('/reports/bezier-cdp-controls.bin')).arrayBuffer()),expected=new Int32Array(await(await fetch('/reports/bezier-cdp-counts.bin')).arrayBuffer());
 const options={objectHeap:'persistent',deviceHeap:{maxAllocations:256,maxElements:32},deviceLaunchQueue:{maxLaunches:256},workgroupSize:[64]};
 const parent=await runtime.kernel(source,{...options,entry:'computeBezierLinesCDP'}),free=await runtime.kernel(source,{...options,entry:'freeVertexMem'}),arena=runtime.createObjectArena();
 const data=new ArrayBuffer(256*32),f=new Float32Array(data);for(let i=0;i<256;i++)f.set(controls.subarray(i*6,i*6+6),i*8);
 const bLines=runtime.createBuffer(new Uint8Array(data));
 try{
  let rejected=false;try{parent.bind({bLines},{nLines:256},{objectArena:arena});}catch(e){rejected=/queueOnly/.test(e.message);}if(!rejected)throw Error('Queue producer was accepted as a complete launch');
  runtime.batch().dispatch(parent.bind({bLines},{nLines:256},{objectArena:arena,queueOnly:true}),[4]).submit();
  const records=await runtime.read(bLines,Uint32Array),types=parent.artifact.metadata.objectHeap.types,q=parent.artifact.metadata.deviceLaunchQueue.queues[0];
  const queueBuffer=arena.buffers[types.findIndex(t=>t.name===q.name)],heapBuffer=arena.buffers[types.findIndex(t=>t.name.startsWith('device_'))],words=await runtime.read(queueBuffer,Uint32Array),seen=new Set();
  if(words[0]!==256||words[1]!==0)throw Error('Unexpected queue count or overflow');
  for(let i=0;i<256;i++){
   const base=4+i*q.stride,index=words[base+3],count=words[base+4];
   if(index>=256||seen.has(index)||count!==expected[index]||records[index*8+7]!==count||records[index*8+6]===0||words[base]!==Math.ceil(count/32)||words[base+1]!==1||words[base+2]!==1)throw Error('Original parent launch mismatch '+i);
   seen.add(index);
  }
  runtime.batch().dispatch(free.bind({bLines},{nLines:256},{objectArena:arena}),[4]).submit();if((await runtime.read(heapBuffer,Uint32Array)).slice(0,256).some(Boolean))throw Error('Original parent allocations leaked');
  const tinySource='__global__ void child(float* out,int i,float value){out[i]=value;} __global__ void parent(float* out,float grid){int i=int(threadIdx.x);child<<<grid,1>>>(out,i,float(i)*0.25f-3.5f);}';
  const tiny=await runtime.kernel(tinySource,{entry:'parent',workgroupSize:[8],objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:2}}),smallArena=runtime.createObjectArena(),out=runtime.createBuffer(32);
  try{
   runtime.batch().dispatch(tiny.bind({out},{grid:1},{objectArena:smallArena,queueOnly:true}),[1]).submit();const small=await runtime.read(smallArena.buffers[0],Uint32Array),floats=new Float32Array(small.buffer);
   if(small[0]!==8||small[1]!==1)throw Error('Queue overflow was not recorded');const ids=new Set();for(let slot=0;slot<2;slot++){const queue=tiny.artifact.metadata.deviceLaunchQueue.queues[0],base=4+slot*queue.stride,index=small[base+3];if(index>=8||ids.has(index)||floats[base+4]!==index*.25-3.5)throw Error('Queue argument snapshot mismatch');ids.add(index);}
   runtime.batch().clear(smallArena.buffers[0]).dispatch(tiny.bind({out},{grid:0.5},{objectArena:smallArena,queueOnly:true}),[1]).submit();const invalid=await runtime.read(smallArena.buffers[0],Uint32Array);if(invalid[0]!==0||invalid[1]!==1)throw Error('Invalid child grid was silently accepted');
  }finally{smallArena.dispose();runtime.destroyBuffer(out);}
  return {curves:256,queuedLaunches:256,countsMatchNative:true,originalParentBody:true,allocationAndCurvatureOnGpu:true,queueOverflow:false,overflowCaseVerified:true,argumentSnapshotsVerified:true,invalidGridRejected:true,allFreed:true,childrenExecuted:false};
 }finally{arena.dispose();runtime.destroyBuffer(bLines);}
}
