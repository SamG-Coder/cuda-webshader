// Optional launch contract: selected linear bounds must end on a block boundary.
export function uniformBlockGuards(kernel,options,walk,fail){
 const requested=options.fullWorkgroups;if(requested===undefined)return [];
 if(!Array.isArray(requested)||!requested.length||requested.some(n=>typeof n!=='string')||new Set(requested).size!==requested.length)fail('fullWorkgroups must name distinct unsigned bounds.');
 const size=options.workgroupSize||[128,1,1];
 if((size[1]??1)!==1||(size[2]??1)!==1)fail('Full-workgroup guards require one-dimensional blocks.');
 const member=(n,base)=>n?.kind==='member'&&n.member==='x'&&n.base.kind==='id'&&n.base.name===base;
 const linear=n=>n?.kind==='binary'&&n.op==='+'&&member(n.right,'threadIdx')&&n.left.kind==='binary'&&n.left.op==='*'&&member(n.left.left,'blockIdx')&&member(n.left.right,'blockDim');
 const constraints=[];
 for(const name of requested){
  if(!kernel.params.some(p=>p.name===name&&!p.pointer&&p.type==='u32'))fail('Full-workgroup bounds must be unsigned scalar kernel parameters.');
  // Reject mutations, including a possible by-reference helper call.
  walk(kernel.body,n=>{if((n.kind==='assign'&&n.left.kind==='id'&&n.left.name===name)||(n.kind==='unary'&&['++','--'].includes(n.op)&&n.value.kind==='id'&&n.value.name===name)||(n.kind==='call'&&n.args.some(a=>a.kind==='id'&&a.name===name))||(['decl','thread-block'].includes(n.kind)&&n.name===name))fail('Full-workgroup bounds must remain unchanged and cannot be passed to helpers.');});
  let found=false;
  for(let i=1;i<kernel.body.body.length;i++){
   const guard=kernel.body.body[i],index=kernel.body.body[i-1],c=guard.condition;
   const ret=guard.yes?.kind==='block'&&guard.yes.body.length===1?guard.yes.body[0]:guard.yes;
   if(index.kind!=='decl'||index.pointer||index.dimensions.length||!['i32','u32'].includes(index.type)||!linear(index.init)||guard.kind!=='if'||guard.no||ret?.kind!=='return'||ret.value||c?.kind!=='binary'||c.op!=='>='||c.left.kind!=='id'||c.left.name!==index.name||c.right.kind!=='id'||c.right.name!==name)continue;
   const token=c.token,dimension=base=>({kind:'member',token,base:{kind:'id',token,name:base},member:'x'});
   guard.condition={kind:'binary',token,op:'>=',left:dimension('blockIdx'),right:{kind:'binary',token,op:'/',left:{kind:'id',token,name},right:dimension('blockDim')}};
   found=true;
  }
  if(!found)fail(`No supported linear early-return guard found for '${name}'.`);
  constraints.push({name,multipleOf:size[0],minimum:0});
 }
 return constraints;
}
