// Complete original parent and child execution; native outputs are comparison data only.
export async function checkBezierScheduled(runtime){
 const original=await(await fetch('/tests/bezier-cdp-device.cuh')).text();
 const source=original+`
 __global__ void allocate_test_lines(BezierLine* lines){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;cudaMalloc((void**)&lines[i].vertexPos,lines[i].nVertices*sizeof(float2));}
 __global__ void capture_test_lines(BezierLine* lines,float2* out){unsigned i=blockIdx.x,j=threadIdx.x;if(j<(unsigned int)lines[i].nVertices)out[i*32u+j]=lines[i].vertexPos[j];}`;
 const controls=new Float32Array(await(await fetch('/reports/bezier-cdp-controls.bin')).arrayBuffer()),counts=new Int32Array(await(await fetch('/reports/bezier-cdp-counts.bin')).arrayBuffer()),expected=new Float32Array(await(await fetch('/reports/bezier-cdp-native.bin')).arrayBuffer());
 const data=new ArrayBuffer(256*32),f=new Float32Array(data),u=new Uint32Array(data);for(let i=0;i<256;i++){f.set(controls.subarray(i*6,i*6+6),i*8);}
 const options={objectHeap:'persistent',deviceHeap:{maxAllocations:256,maxElements:32},deviceLaunchQueue:{maxLaunches:256}},kernels={};
 for(const entry of ['computeBezierLinesCDP','capture_test_lines','freeVertexMem'])kernels[entry]=await runtime.kernel(source,{...options,entry,...(entry==='computeBezierLinesCDP'?{scheduleDeviceLaunches:true}:{}),workgroupSize:['computeBezierLinesCDP','freeVertexMem'].includes(entry)?[64]:[32]});
 const arena=runtime.createObjectArena(),records=runtime.createBuffer(new Uint8Array(data)),out=runtime.createBuffer(expected.byteLength);
 try{
  await kernels.computeBezierLinesCDP.runQueued({bLines:records},{nLines:256},[4],{objectArena:arena});
  const produced=await runtime.read(records,Uint32Array);for(let i=0;i<256;i++)if(produced[i*8+7]!==counts[i])throw Error('Scheduled parent count mismatch '+i);
  runtime.batch().dispatch(kernels.capture_test_lines.bind({lines:records,out},{},{objectArena:arena}),[256]).submit();const actual=await runtime.read(out);let maxError=0;
  for(let i=0;i<actual.length;i++){const e=Math.abs(actual[i]-expected[i]);if(!Number.isFinite(actual[i])||e>1e-6)throw Error('Original Bezier child mismatch '+i);maxError=Math.max(maxError,e);}
  runtime.batch().dispatch(kernels.freeVertexMem.bind({bLines:records},{nLines:256},{objectArena:arena}),[4]).submit();const slots=await runtime.read(arena.buffers[kernels.freeVertexMem.artifact.metadata.objectHeap.types.findIndex(t=>t.name.startsWith('device_'))],Uint32Array);if(slots.slice(0,256).some(Boolean))throw Error('Original free kernel leaked vertex buffers');
  const tinySource='__global__ void child(float* out,int i,float v){out[i]=v+float(blockIdx.x*blockDim.x+threadIdx.x);} __global__ void parent(float* out,int n){int i=int(threadIdx.x);if(i<n)child<<<1,1>>>(out,i,float(i)*.25f-3.5f);}';
  const tiny=await runtime.kernel(tinySource,{entry:'parent',workgroupSize:[8],objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:16},scheduleDeviceLaunches:true}),smallArena=runtime.createObjectArena(),values=runtime.createBuffer(32);
  try {
    for(const n of [8,3,0]){
      runtime.batch().clear(values).submit();const before=runtime.stats.readbackBytes;
      await tiny.runQueued({out:values},{n},[1],{objectArena:smallArena});
      if(runtime.stats.readbackBytes-before!==8)throw Error('Scheduler read back data beyond the error header');
      const actual=await runtime.read(values);for(let i=0;i<8;i++)if(actual[i]!== (i<n?i*.25-3.5:0))throw Error('Scheduled argument or stale queue mismatch');
    }
    const overflow=await runtime.kernel(tinySource,{entry:'parent',workgroupSize:[8],objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:2},scheduleDeviceLaunches:true}),overflowArena=runtime.createObjectArena();
    try{let rejected=false;try{await overflow.runQueued({out:values},{n:8},[1],{objectArena:overflowArena});}catch(e){rejected=/overflow/.test(e.message);}if(!rejected)throw Error('Scheduled overflow did not reject');}finally{overflowArena.dispose();}
  }finally{smallArena.dispose();runtime.destroyBuffer(values);}
  return {queueReuseVerified:true,unusedSlotsSkipped:true,overflowRejected:true,schedulingReadback:'error header only',curves:256,vertices:counts.reduce((a,b)=>a+b,0),maxError,originalVertexAndFreeBodies:true,allFreed:true,parentSchedulingSupported:true,launchCountsFromNativeFixture:false,indirectDispatch:true};
 }finally{arena.dispose();runtime.destroyBuffer(records);runtime.destroyBuffer(out);}
}
