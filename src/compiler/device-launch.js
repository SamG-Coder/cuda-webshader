// Explicit producer stage for a bounded GPU child-launch queue.
// Queue production alone does not execute the queued child work.
export function launchQueues(e,walk,constantValue){
 const options=e.options.deviceLaunchQueue;if(options===undefined)return [];
 if(!options||typeof options!=='object'||Array.isArray(options)||Object.keys(options).some(k=>k!=='maxLaunches')||!Number.isInteger(options.maxLaunches)||options.maxLaunches<1||options.maxLaunches>65535)e.fail('deviceLaunchQueue requires maxLaunches in 1..65535.',e.kernel);
 if(!e.persistentObjects)e.fail('Device launch queues require objectHeap: persistent.',e.kernel);
 const queues=[];
 for(const caller of e.ast.functions){walk(caller.body,n=>{
  if(n.kind!=='device-launch')return;
  if(caller.qualifier!=='__global__')e.fail('Device launches in helpers are not supported yet.',n);
  const child=e.ast.functions.find(f=>f.name===n.callee.name&&f.qualifier==='__global__');if(!child)e.fail('Child launch must name a declared global kernel.',n);
  if(n.configuration.length!==2||child.params.length!==n.args.length)e.fail('Child queues require grid and block dimensions and every argument.',n);
  let block;try{block=constantValue(n.configuration[1]);}catch{}if(!Number.isInteger(block)||block<1||block>1024)e.fail('Child block size must be a constant in 1..1024.',n);
  const scalars=[],buffers=[];for(let i=0;i<child.params.length;i++){
   const p=child.params[i],arg=n.args[i];
   if(p.pointer){const parent=caller.params.find(x=>arg.kind==='id'&&x.name===arg.name);if(!parent?.pointer||p.type!==parent.type)e.fail('Child buffer arguments must be matching named parent buffers.',arg);buffers.push({name:p.name,parent:parent.name,type:p.type});}
   else{if(p.reference||!['i32','u32','f32'].includes(p.type))e.fail('Child queue scalar arguments require 32-bit integer or float values.',p);scalars.push({name:p.name,type:p.type,argument:i,word:3+scalars.length});}
  }
  const id=queues.length,stride=3+scalars.length,queue={id,name:'launch_queue_'+id,caller:caller.name,child:child.name,capacity:options.maxLaunches,block:[block,1,1],stride,scalars,buffers,binding:e.objectHeaps.size+e.objectImports.length+e.deviceHeaps.size+id,byteLength:16+options.maxLaunches*stride*4,variable:'cw_launch_queue_'+id};
  queue.recordLayout=JSON.stringify({caller:queue.caller,child:queue.child,capacity:queue.capacity,stride,scalars,buffers,block:queue.block});n.queueId=id;queues.push(queue);
 });}
 return queues;
}
export function launchQueueDeclarations(queues){return queues.flatMap(q=>[
 `struct CWLaunchQueue_${q.id} { count: atomic<u32>, overflow: atomic<u32>, pad0: u32, pad1: u32, words: array<u32, ${q.capacity*q.stride}>, }`,
 `@group(1) @binding(${q.binding}) var<storage,read_write> ${q.variable}: CWLaunchQueue_${q.id};`
]);}
export function emitLaunch(e,n){
 const q=e.launchQueues.find(q=>q.id===n.queueId);if(!q)e.fail('Device child-kernel launches require GPU scheduling support.',n);
 const grid=e.expr(n.configuration[0]);if(!['i32','u32','f32'].includes(grid.type))e.fail('Child grid must be a one-dimensional scalar.',n);
 const name='cw_launch_'+e.temp++,pre=[...grid.pre,`let ${name}_grid=${grid.code};`],values=[];
 for(const scalar of q.scalars){const value=e.expr(n.args[scalar.argument]),code=e.convert(value.code,value.type,scalar.type,n.args[scalar.argument]),local=name+'_'+scalar.word;pre.push(...value.pre,`let ${local}=${code};`);values.push({word:scalar.word,code:scalar.type==='u32'?local:`bitcast<u32>(${local})`});}
 const max=grid.type==='f32'?'65535.0f':grid.type==='u32'?'65535u':'65535i',minimum=grid.type==='f32'?'1.0f':grid.type==='u32'?'1u':'1i';
 pre.push(`if(${name}_grid>=${minimum} && ${name}_grid<=${max}) {`,`let ${name}_slot=atomicAdd(&${q.variable}.count,1u);`,`if(${name}_slot<${q.capacity}u) {`,`let ${name}_base=${name}_slot*${q.stride}u;`,`${q.variable}.words[${name}_base]=u32(${name}_grid);`,`${q.variable}.words[${name}_base+1u]=1u;`,`${q.variable}.words[${name}_base+2u]=1u;`,...values.map(v=>`${q.variable}.words[${name}_base+${v.word}u]=${v.code};`),`} else { atomicStore(&${q.variable}.overflow,1u); }`,`} else { atomicStore(&${q.variable}.overflow,1u); }`);
 return e.result(n,'void','',pre);
}
