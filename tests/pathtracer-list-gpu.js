export async function checkPersistentList(runtime){
 const source=(await Promise.all(['value-class','list-class','list-kernels'].map(async n=>await(await fetch('/tests/pathtracer-'+n+'.cuh')).text()))).join('\n');
 const kernels=[];for(const entry of ['create_list','update_list','trace_list','free_list'])kernels.push(await runtime.kernel(source,{entry,workgroupSize:[entry==='trace_list'?2:1,1,1],objectHeap:'persistent'}));
 const arena=runtime.createObjectArena(),objects=runtime.createBuffer(8),world=runtime.createBuffer(4),out=runtime.createBuffer(16);
 const expected=new Float32Array(await(await fetch('/reports/pathtracer-list-native.bin')).arrayBuffer());let compared=0;
 try{
  // Creation and updates use a single invocation; tracing covers hit and miss.
  runtime.batch().dispatch(kernels[0].bind({objects,world},{},{objectArena:arena}),[1,1,1]).submit();await runtime.idle();
  for(let cycle=0;cycle<2;cycle++){
   if(cycle){runtime.batch().dispatch(kernels[1].bind({},{},{objectArena:arena}),[1,1,1]).submit();await runtime.idle();}
   runtime.batch().dispatch(kernels[2].bind({out},{},{objectArena:arena}),[1,1,1]).submit();
   const actual=await runtime.read(out);for(let i=0;i<4;i++){if(actual[i]!==expected[cycle*4+i])throw Error('List mismatch '+cycle+':'+i+' '+actual[i]+' vs '+expected[cycle*4+i]);compared++;}
  }
  runtime.batch().dispatch(kernels[3].bind({},{},{objectArena:arena}),[1,1,1]).submit();await runtime.idle();
  if((await runtime.read(objects,Uint32Array)).some(Boolean)||(await runtime.read(world,Uint32Array)).some(Boolean))throw Error('List cleanup failed');
  return {values:compared,nativeExact:true,capturedBufferMutation:true,missPreserved:true,fullPathtracerSupported:false};
 }finally{arena.dispose();for(const b of [objects,world,out])runtime.destroyBuffer(b);}
}
