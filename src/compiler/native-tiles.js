// A CUDA tile maps to an explicitly sized WGSL subgroup. Logical CUDA thread
// indices are remapped from subgroup IDs, never assumed to equal physical IDs.
export function lowerNativeTiles(ast,walk,fail){
 for(const fn of ast.functions){const handles=new Set(),blocks=new Set();walk(fn.body,n=>{if(n.kind==='thread-block')blocks.add(n.name);});
  walk(fn.body,n=>{if(n.kind==='thread-warp'){if(fn.qualifier!=='__global__'||!blocks.has(n.parent))fail('Static tiles require a kernel-local thread block.',n);if(handles.has(n.name))fail('Static tile names must be unique.',n);handles.add(n.name);fn.nativeTiles=true;n.kind='empty';}});
  if(!handles.size)continue;
  walk(fn.body,n=>{if(n.kind==='decl'&&handles.has(n.name))fail('Static tile handles cannot be shadowed.',n);
   if(n.kind==='call'&&n.callee.kind==='member'&&n.callee.base.kind==='id'&&handles.has(n.callee.base.name)){const method=n.callee.member;if(!['thread_rank','any','ballot','shfl','shfl_up'].includes(method))fail('Unsupported static tile operation '+method,n);n.callee={kind:'id',name:'cw_native_tile_'+method,token:n.token};}
   if(n.kind==='call'&&n.callee.name==='cooperative_groups::sync'&&n.args.some(a=>handles.has(a.name)))fail('Static tile memory synchronization requires predicated workgroup phases.',n);
  });
  // The vote itself makes the loop decision identical within a tile. Some
  // validators retain its predicate's lane dependence around the back-edge.
  // Restrict the diagnostic override to loops with no divergent collectives
  // or early exits, and leave an unsuppressed entry collective outside.
  const collective=n=>n?.kind==='call'&&n.callee?.name?.startsWith('cw_native_tile_')&&n.callee.name!=='cw_native_tile_thread_rank';
  walk(fn.body,loop=>{if(!['for','while'].includes(loop.kind)||loop.condition?.callee?.name!=='cw_native_tile_any')return;
    const prove=(node,conditional=false)=>{if(!node||typeof node!=='object')return;if(Array.isArray(node)){node.forEach(n=>prove(n,conditional));return;}
      if(['break','continue','return'].includes(node.kind))fail('Vote-controlled tile loops cannot exit or continue early.',node);
      if(collective(node)&&conditional)fail('Vote-controlled tile loops require unconditional collectives.',node);
      const next=conditional||['if','for','while','do','conditional'].includes(node.kind)||node.kind==='binary'&&['&&','||'].includes(node.op);
      for(const [key,value]of Object.entries(node))if(key!=='token')prove(value,next);
    };prove(loop.body);prove(loop.step);prove(loop.condition.args);loop.provenTileVoteLoop=true;
  });

 }
}
export function emitNativeTile(e,n,name){
 const op=name.slice('cw_native_tile_'.length),args=n.args.map(a=>e.expr(a));
 const count=op==='thread_rank'?0:['any','ballot'].includes(op)?1:2;if(args.length!==count)e.fail('Wrong static tile argument count.',n);
 if(op==='thread_rank')return e.result(n,'u32','(cw_thread.x % 32u)');
 if(['any','ballot'].includes(op)){const a=args[0],predicate=e.convert(a.code,a.type,'bool',n);return e.result(n,op==='any'?'bool':'u32',op==='any'?`subgroupAny(${predicate})`:`subgroupBallot(${predicate}).x`,a.pre);}
 const [value,index]=args;if(!['i32','u32','f32'].includes(value.type)||!['i32','u32'].includes(index.type))e.fail('Tile shuffle requires a scalar value and integer lane.',n);
 const temp='cw_shuffle_'+e.temp++,pre=[...value.pre,`let ${temp}=${value.code};`,...index.pre];
 if(op==='shfl')return e.result(n,value.type,`subgroupShuffle(${temp}, u32(${index.code}) % 32u)`,pre);
 const delta=temp+'_delta',shuffled=temp+'_value';pre.push(`let ${delta}=u32(${index.code});`,`let ${shuffled}=subgroupShuffle(${temp}, select(0u,(cw_thread.x % 32u)-${delta},(cw_thread.x % 32u)>=${delta}));`);
 return e.result(n,value.type,`select(${temp},${shuffled},(cw_thread.x % 32u)>=${delta})`,pre);
}
