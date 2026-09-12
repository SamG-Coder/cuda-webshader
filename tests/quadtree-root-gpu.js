export async function checkQuadtreeRoot(runtime){
 const source=(await Promise.all(['quadtree-cdp-device.cuh','quadtree-cdp-setup.cuh'].map(async f=>(await fetch('/tests/'+f)).text()))).join('\n');
 const kernel=await runtime.kernel(source,{entry:'build_quadtree_kernel<128>',valueBuffers:['nodes','points'],workgroupSize:[128],sharedMemoryBytes:64,objectHeap:'persistent',deviceHeap:{maxAllocations:1024,maxElements:1024},deviceLaunchQueue:{maxLaunches:1024}}),meta=kernel.artifact.metadata;
 const input=new Float32Array(await(await fetch('/reports/quadtree-input.bin')).arrayBuffer()),expected=(await(await fetch('/reports/quadtree-nodes.json')).json()).filter(n=>n.depth<=1),n=input.length/2;
 // Host-equivalent root allocation and captured coordinate descriptors.
 if(meta.bindings.find(b=>b.name==='nodes').stride!==32||meta.bindings.find(b=>b.name==='points').stride!==8)throw Error('Quadtree storage layout changed');
 const nodeBytes=new ArrayBuffer(21845*32),view=new DataView(nodeBytes);view.setFloat32(16,1,true);view.setFloat32(20,1,true);view.setInt32(28,n,true);
 const imports={x0:runtime.createBuffer(Float32Array.from({length:n},(_,i)=>input[2*i])),y0:runtime.createBuffer(Float32Array.from({length:n},(_,i)=>input[2*i+1])),x1:runtime.createBuffer(n*4),y1:runtime.createBuffer(n*4)};
 const points=runtime.createBuffer(Uint32Array.from(['x0','y0','x1','y1'],name=>meta.objectHeap.imports.find(d=>d.name===name).id*1048576)),nodes=runtime.createBuffer(new Uint8Array(nodeBytes)),arena=runtime.createObjectArena();
 try{
  runtime.batch().dispatch(kernel.bind({nodes,points,...imports},{'params.point_selector':0,'params.num_nodes_at_this_level':1,'params.depth':0,'params.max_depth':8,'params.min_points_per_node':16},{objectArena:arena,queueOnly:true}),[1]).submit();
  const actual=new DataView((await runtime.read(nodes,Uint32Array,5*32)).buffer),x=await runtime.read(imports.x1),y=await runtime.read(imports.y1);
  let fields=0;for(const node of expected){const base=node.index*32,values=[actual.getInt32(base,true),actual.getInt32(base+24,true),actual.getInt32(base+28,true),...[8,12,16,20].map(offset=>actual.getFloat32(base+offset,true))],reference=[node.id,node.begin,node.end,...node.bounds];if(values.join(',')!==reference.join(','))throw Error('Native quadtree root node mismatch '+node.index+': '+values+' expected '+reference);fields+=values.length;}
  const before=Array.from({length:n},(_,i)=>input[2*i]+','+input[2*i+1]).sort(),after=Array.from({length:n},(_,i)=>x[i]+','+y[i]).sort();if(before.join(';')!==after.join(';'))throw Error('Quadtree root lost or duplicated points');
  for(const node of expected.filter(node=>node.depth===1))for(let i=node.begin;i<node.end;i++)if(x[i]<node.bounds[0]||x[i]>=node.bounds[2]||y[i]<node.bounds[1]||y[i]>=node.bounds[3])throw Error('Quadtree point outside native child bounds');
  const q=meta.deviceLaunchQueue.queues.find(q=>q.caller===kernel.artifact.name),words=await runtime.read(arena.buffers[meta.objectHeap.types.findIndex(t=>t.name===q.name)],Uint32Array,16+q.stride*4);
  if(words[0]!==1||words[1]!==0||words[4]!==4)throw Error('Quadtree root did not enqueue its four children');
  const queued={};for(const s of q.scalars)queued[s.name]=words[4+s.word];for(const b of q.buffers)queued[b.name]=words[4+b.offsetWord];
  if(queued['params.depth']!==1||queued['params.point_selector']!==1||queued['params.num_nodes_at_this_level']!==4||queued.nodes!==1||queued.points!==0)throw Error('Quadtree child arguments mismatch');
  let rejected=0;for(const invalid of [source.replace('nodes[blockIdx.x]','nodes[threadIdx.x]'),source.replace('int num_points = node.num_points();','if(threadIdx.x==0)return;int num_points = node.num_points();')]){
   try{await runtime.kernel(invalid,{entry:'build_quadtree_kernel<128>',valueBuffers:['nodes','points'],workgroupSize:[128],sharedMemoryBytes:64,objectHeap:'persistent',deviceHeap:{maxAllocations:1024,maxElements:1024},deviceLaunchQueue:{maxLaunches:1024}});}catch(e){if(/uniform control flow/.test(e.message)){rejected++;continue;}throw e;}throw Error('Non-uniform quadtree control flow was accepted');
  }
  return {originalKernel:true,points:n,nodesCompared:expected.length,fieldsCompared:fields,pointPermutationExact:true,allPointsInNativeQuadrants:true,queuedChildBlocks:4,nonUniformVariantsRejected:rejected,recursiveExecution:false};
 }finally{arena.dispose();for(const b of [nodes,points,...Object.values(imports)])runtime.destroyBuffer(b);}
}
