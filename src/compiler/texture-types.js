// Infer a single sampling format for each connected texture-parameter chain.
export function inferTextureTypes(functions,kernel,walk,fail){
 for(const f of functions)if(['texture3d','surface2d'].includes(f.result))fail('Helpers cannot return texture or surface handles.',f);
 const definitions=new Map(functions.map(f=>[f.name,f])),parents=new Map(),uses=[],edges=[];
 const parameters=new Map(functions.map(f=>[f,new Map(f.params.filter(p=>p.type==='texture3d').map(p=>{parents.set(p,p);return [p.name,p];}))]));
 const root=p=>{if(parents.get(p)!==p)parents.set(p,root(parents.get(p)));return parents.get(p);};
 for(const f of functions.filter(f=>f.qualifier!=='__global__'||f===kernel))walk(f.body,n=>{
  if(n.kind!=='call'||n.callee.kind!=='id')return;const name=n.callee.name,params=parameters.get(f);
  if(['tex1D','tex1Dfetch','tex2D','tex3D'].includes(name)){const p=n.args[0]?.kind==='id'?params.get(n.args[0].name):null;if(p)uses.push([p,name==='tex1Dfetch'?'fetch_'+n.callee.templateArgument:name==='tex2D'&&n.callee.templateArgument==='float4'?'tex2Dfloat4':name,n]);return;}
  const target=definitions.get(name);if(!target){if(n.args.some(a=>a.kind==='id'&&params.has(a.name)))fail('Texture helper calls require a statically resolved function; supply explicit template arguments.',n);return;}
  target.params.forEach((p,i)=>{if(p.type!=='texture3d')return;const arg=n.args[i],from=arg?.kind==='id'?params.get(arg.name):null;if(from)edges.push([from,p]);});
 });
 for(const [a,b]of edges)parents.set(root(a),root(b));const sampling=new Map();
 for(const [p,kind,node]of uses){const key=root(p);if(sampling.has(key)&&sampling.get(key)!==kind)fail('A texture parameter cannot mix sampling dimensions or formats through helper calls.',node);sampling.set(key,kind);}
 for(const p of parents.keys())p.textureSampling=sampling.get(root(p))||'tex3D';
}
export function textureShape(sampling){return {dimension:sampling==='tex3D'?'3d':'2d',format:sampling==='fetch_uint'?'r32uint':(sampling==='tex1D'||sampling==='tex2Dfloat4')?'rgba32float':sampling==='tex2D'?'r32float':'r8unorm'};}
