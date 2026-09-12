export async function checkPersistentObjects(runtime){
 const source=await(await fetch('/tests/pathtracer-value-class.cuh')).text()+await(await fetch('/tests/pathtracer-persistent-kernels.cuh')).text();
 const kernels=[];for(const entry of ['create_objects','trace_objects','free_objects'])kernels.push(await runtime.kernel(source,{entry,workgroupSize:[64,1,1],objectHeap:'persistent'}));
 const arena=runtime.createObjectArena(),world=runtime.createBuffer(512*4),out=runtime.createBuffer(512*5*4),expected=new Float32Array(await(await fetch('/reports/pathtracer-persistent-native.bin')).arrayBuffer());let compared=0;
 try{
  for(let cycle=0;cycle<2;cycle++){
   const offset=cycle*3;
   runtime.batch().dispatch(kernels[0].bind({world},{offset},{objectArena:arena}),[8,1,1]).submit();await runtime.idle();
   const pointers=await runtime.read(world,Uint32Array);if(pointers.some(p=>p===0)||new Set(pointers).size!==512)throw Error('Persistent allocation identities are not distinct and non-null.');
   runtime.batch().dispatch(kernels[1].bind({world,out},{offset},{objectArena:arena}),[8,1,1]).submit();const actual=await runtime.read(out);
   for(let i=0;i<actual.length;i++){if(actual[i]!==expected[cycle*actual.length+i])throw Error('Persistent sphere mismatch at '+cycle+':'+i+' '+actual[i]+' vs '+expected[cycle*actual.length+i]);compared++;}
   runtime.batch().dispatch(kernels[2].bind({world},{},{objectArena:arena}),[8,1,1]).submit();const cleared=await runtime.read(world,Uint32Array);if(cleared.some(p=>p!==0))throw Error('Freed object pointers were not cleared.');
  }
  const other=runtime.createObjectArena();try{let rejected=false;try{kernels[1].bind({world,out},{offset:0},{objectArena:other});}catch(error){rejected=/another arena/.test(error.message);}if(!rejected)throw Error('Cross-arena pointer buffer was accepted.');}finally{other.dispose();}
  return {arenaIsolation:true,cycles:2,objectsPerCycle:512,values:compared,nativeExact:true,separateSubmissions:true,allFreed:true,persistent:true,fullPathtracerSupported:false};
 }finally{arena.dispose();runtime.destroyBuffer(world);runtime.destroyBuffer(out);}
}
