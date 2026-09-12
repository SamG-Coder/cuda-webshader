export async function checkDeviceHeap(runtime){
 const source=await(await fetch('/tests/device-heap.cu')).text(),kernels={},options={workgroupSize:[64],objectHeap:'persistent',deviceHeap:{maxAllocations:256,maxElements:32}};
 for(const entry of ['allocate_values','write_values','read_values','free_values'])kernels[entry]=await runtime.kernel(source,{...options,entry});
 const expected=new Float32Array(await(await fetch('/reports/device-heap-native.bin')).arrayBuffer());
 const arena=runtime.createObjectArena(),records=runtime.createBuffer(256*8),status=runtime.createBuffer(256*4),output=runtime.createBuffer(256*32*8);
 const dispatch=(entry,buffers)=>runtime.batch().dispatch(kernels[entry].bind(buffers,{},{objectArena:arena}),[4]).submit();
 try{
  for(let cycle=0;cycle<2;cycle++){
   dispatch('allocate_values',{records,status});if((await runtime.read(status,Int32Array)).some(Boolean))throw Error('Device allocation failed');
   dispatch('write_values',{records});await runtime.idle();dispatch('read_values',{records,output});const actual=await runtime.read(output);
   for(let i=0;i<expected.length;i++)if(actual[i]!==expected[i])throw Error('Native device heap mismatch '+cycle+':'+i);
   dispatch('free_values',{records,status});if((await runtime.read(status,Int32Array)).some(Boolean))throw Error('Device free failed');
   const slots=await runtime.read(arena.buffers[0],Uint32Array);if(slots.slice(0,256).some(Boolean))throw Error('Heap slots leaked');
  }
  const other=runtime.createObjectArena();try{let rejected=false;try{kernels.read_values.bind({records,output},{},{objectArena:other});}catch(e){rejected=/another arena/.test(e.message);}if(!rejected)throw Error('Pointer records escaped arena ownership');}finally{other.dispose();}
  const edgeSource=`struct A {float2* p;};
  __global__ void alloc(A* a,int* result,int slot,unsigned int bytes){result[0]=cudaMalloc((void**)&a[slot].p,bytes);}
  __global__ void release(A* a,int* result,int slot){result[0]=cudaFree(a[slot].p);a[slot].p=NULL;}
  __global__ void bounds(A* a,float2* out){a[1].p[0]=make_float2(3.f,7.f);a[1].p[1]=make_float2(99.f,99.f);a[1].p[-1]=make_float2(88.f,88.f);a[1].p[0].x+=2.f;out[0]=a[1].p[0];out[1]=a[1].p[1];}`;
  const small=runtime.createObjectArena(),a=runtime.createBuffer(8),result=runtime.createBuffer(4),out=runtime.createBuffer(16),edge={};
  try{
   for(const entry of ['alloc','release','bounds'])edge[entry]=await runtime.kernel(edgeSource,{entry,workgroupSize:[1],objectHeap:'persistent',deviceHeap:{maxAllocations:1,maxElements:2}});
   const run=async(entry,scalars={},want=0)=>{runtime.batch().dispatch(edge[entry].bind(entry==='bounds'?{a,out}:{a,result},scalars,{objectArena:small}),[1]).submit();if(entry!=='bounds'&&(await runtime.read(result,Int32Array))[0]!==want)throw Error('Device heap edge status mismatch '+entry);};
   await run('alloc',{slot:0,bytes:0});await run('release',{slot:0});
   await run('alloc',{slot:0,bytes:8});await run('alloc',{slot:1,bytes:8},2);
   await run('release',{slot:0});await run('alloc',{slot:1,bytes:8});await run('bounds');
   if(JSON.stringify([...await runtime.read(out)])!=='[5,7,0,0]')throw Error('Allocation bounds isolation failed');
   await run('release',{slot:1});await run('alloc',{slot:0,bytes:24},2);await run('alloc',{slot:0,bytes:7},2);
  }finally{small.dispose();for(const b of [a,result,out])runtime.destroyBuffer(b);}
  return {allocationsPerCycle:256,cycles:2,liveElementsPerCycle:4224,floatComponentsCompared:32768,nativeExact:true,allFreed:true,reused:true,arenaIsolation:true,capacityFailure:true,zeroByteAllocation:true,boundsIsolation:true};
 }finally{arena.dispose();for(const b of [records,status,output])runtime.destroyBuffer(b);}
}
