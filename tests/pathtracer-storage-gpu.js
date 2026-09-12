export async function checkPathtracerStorage(runtime){
 const source=(await Promise.all(['/tests/pathtracer-material-class.cuh','/tests/pathtracer-storage-class.cuh','/tests/pathtracer-storage-kernels.cuh'].map(async p=>await(await fetch(p)).text()))).join('\n');
 const kernels={};for(const entry of ['render_init','seed_wide','draw_states','create_storage_scene','free_storage_scene'])kernels[entry]=await runtime.kernel(source,{entry,valueBuffers:['fb'],libraries:['curand-xorwow'],objectHeap:'persistent',workgroupSize:entry==='render_init'?[8,8,1]:[64,1,1]});
 const stride=kernels.render_init.artifact.metadata.bindings.find(b=>b.name==='rand_state').stride;if(stride!==24)throw Error('Unexpected RNG record stride');
 if(kernels.draw_states.artifact.metadata.bindings.find(b=>b.name==='fb').stride!==12)throw Error('Unexpected vec3 record stride');
 const arena=runtime.createObjectArena(),rand_state=runtime.createBuffer(512*stride),fb=runtime.createBuffer(512*12),bits=runtime.createBuffer(512*4),world=runtime.createBuffer(512*4),resources={rand_state,fb,bits,world};
 const expected=new Float32Array(await(await fetch('/reports/pathtracer-storage-native.bin')).arrayBuffer()),expectedBits=new Uint32Array(await(await fetch('/reports/pathtracer-storage-native-bits.bin')).arrayBuffer());
 const dispatch=(entry,buffers,scalars={},grid=[8,1,1])=>runtime.batch().dispatch(kernels[entry].bind(buffers,scalars,{objectArena:arena}),grid).submit();
 try{dispatch('render_init',{rand_state},{max_x:32,max_y:16},[4,2,1]);await runtime.idle();
 for(let cycle=0;cycle<3;cycle++){if(cycle===2){dispatch('seed_wide',{rand_state});await runtime.idle();}dispatch('draw_states',{rand_state,fb,bits});const actual=await runtime.read(fb),words=await runtime.read(bits,Uint32Array);for(let i=0;i<1536;i++)if(actual[i]!==expected[cycle*1536+i])throw Error('RNG framebuffer mismatch '+cycle+':'+i+' '+actual[i]+' vs '+expected[cycle*1536+i]);for(let i=0;i<512;i++)if(words[i]!==expectedBits[cycle*512+i])throw Error('RNG words mismatch '+cycle+':'+i);}
 dispatch('create_storage_scene',{world});await runtime.idle();dispatch('free_storage_scene',{world});await runtime.idle();if((await runtime.read(world,Uint32Array)).some(Boolean))throw Error('World pointers not cleared');for(const b of arena.buffers)if((await runtime.read(b,Uint32Array)).slice(0,1024).some(Boolean))throw Error('Object pool still contains live allocations');
 return {floatValues:4608,randomWords:1536,nativeExact:true,wideSeeds:true,recordStride:stride,framebufferStride:12,freedSphereMaterialPairs:512,allPoolsEmpty:true,fullPathtracerSupported:false};
 }finally{arena.dispose();for(const r of Object.values(resources))runtime.destroyBuffer(r);}
}
