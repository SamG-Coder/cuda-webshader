// Stage verification only. The original parent still needs a GPU launch scheduler.
export async function checkBezierChild(runtime){
 const original=await(await fetch('/tests/bezier-cdp-device.cuh')).text();
 const source=original+`
 __global__ void allocate_test_lines(BezierLine* lines){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;cudaMalloc((void**)&lines[i].vertexPos,lines[i].nVertices*sizeof(float2));}
 __global__ void capture_test_lines(BezierLine* lines,float2* out){unsigned i=blockIdx.x,j=threadIdx.x;if(j<(unsigned int)lines[i].nVertices)out[i*32u+j]=lines[i].vertexPos[j];}`;
 const controls=new Float32Array(await(await fetch('/reports/bezier-cdp-controls.bin')).arrayBuffer()),counts=new Int32Array(await(await fetch('/reports/bezier-cdp-counts.bin')).arrayBuffer()),expected=new Float32Array(await(await fetch('/reports/bezier-cdp-native.bin')).arrayBuffer());
 const data=new ArrayBuffer(256*32),f=new Float32Array(data),u=new Uint32Array(data);for(let i=0;i<256;i++){f.set(controls.subarray(i*6,i*6+6),i*8);u[i*8+7]=counts[i];}
 const options={objectHeap:'persistent',deviceHeap:{maxAllocations:256,maxElements:32}},kernels={};
 for(const entry of ['allocate_test_lines','computeBezierLinePositions','capture_test_lines','freeVertexMem'])kernels[entry]=await runtime.kernel(source,{...options,entry,workgroupSize:['allocate_test_lines','freeVertexMem'].includes(entry)?[64]:[32]});
 const arena=runtime.createObjectArena(),records=runtime.createBuffer(new Uint8Array(data)),out=runtime.createBuffer(expected.byteLength);
 try{
  runtime.batch().dispatch(kernels.allocate_test_lines.bind({lines:records},{},{objectArena:arena}),[4]).submit();await runtime.idle();
  const batch=runtime.batch();for(let i=0;i<256;i++)batch.dispatch(kernels.computeBezierLinePositions.bind({bLines:records},{lidx:i,nTessPoints:counts[i]},{objectArena:arena}),[1]);batch.submit();await runtime.idle();
  runtime.batch().dispatch(kernels.capture_test_lines.bind({lines:records,out},{},{objectArena:arena}),[256]).submit();const actual=await runtime.read(out);let maxError=0;
  for(let i=0;i<actual.length;i++){const e=Math.abs(actual[i]-expected[i]);if(!Number.isFinite(actual[i])||e>1e-6)throw Error('Original Bezier child mismatch '+i);maxError=Math.max(maxError,e);}
  runtime.batch().dispatch(kernels.freeVertexMem.bind({bLines:records},{nLines:256},{objectArena:arena}),[4]).submit();const slots=await runtime.read(arena.buffers[0],Uint32Array);if(slots.slice(0,256).some(Boolean))throw Error('Original free kernel leaked vertex buffers');
  return {curves:256,vertices:counts.reduce((a,b)=>a+b,0),maxError,originalVertexAndFreeBodies:true,allFreed:true,parentSchedulingSupported:false,launchCountsFromNativeFixture:true};
 }finally{arena.dispose();runtime.destroyBuffer(records);runtime.destroyBuffer(out);}
}
