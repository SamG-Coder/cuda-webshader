import {KERNELS} from '../kernels.js';
export function suggestLaunch(source){
 const name=source.match(/__global__\s+void\s+(\w+)/)?.[1]||'';
 const known=KERNELS.find(k=>k.id===name);
 return {entry:name,block:known?.workgroupSize||(/threadIdx\.y/.test(source)?[8,8,1]:[128,1,1])};
}
export function suggestConfig(artifact){
 const {bindings,scalars,workgroupSize:block}=artifact.metadata;
 const values=Object.fromEntries(scalars.map(s=>[s.name,s.origin==='constant'?s.defaultValue:({n:65536,n4:16384,width:256,height:256,M:64,N:64,K:64,a:0.5,dt:0.016,time:0,attraction:1})[s.name]??(s.type==='f32'?1:256)]));
 const n=values.n??(values.n4?values.n4*4:(values.width??256)*(values.height??256));
 let groups=[Math.ceil(n/block[0]),1,1];
 if(block[1]>1){const tile=artifact.name==='transpose'?32:artifact.name.startsWith('matmul_')&&artifact.name!=='matmul_naive'?16:block[0];groups=[Math.ceil((values.width??values.N??256)/tile),Math.ceil((values.height??values.M??256)/(artifact.name==='transpose'?32:tile)),1];}
 if(artifact.name==='reduce_sum')groups=[Math.ceil(n/256),1,1];
 if(artifact.name==='histogram')groups=[256,1,1];
 const buffers=Object.fromEntries(bindings.map(b=>{
  let records=values.n4??n;
  if(b.name==='A'&&values.M!==undefined)records=values.M*values.K;
  if(b.name==='B'&&values.N!==undefined)records=values.K*values.N;
  if(b.name==='C'&&values.M!==undefined)records=values.M*values.N;
  if(b.name==='bins')records=256;
  if(artifact.name==='reduce_sum'&&b.name==='output')records=groups[0];
  return [b.name,{records,fill:b.name==='position'?'sphere':b.readOnly?'random':'zero'}];
 }));
 return {groups,scalars:values,buffers,output:(bindings.find(b=>!b.readOnly&&b.elementType==='vec4<f32>')||bindings.find(b=>!b.readOnly)||bindings[0])?.name};
}
export function validateConfig(config,metadata){
 if(!Array.isArray(config.groups)||config.groups.length!==3||config.groups.some(n=>!Number.isInteger(n)||n<1||n>65535)||config.groups.reduce((a,b)=>a*b,1)*metadata.workgroupSize.reduce((a,b)=>a*b,1)>4194304)throw Error('Launch must contain three positive block counts, each ≤65,535, and at most 4,194,304 invocations.');
 if(!config.scalars||!config.buffers)throw Error('Settings need scalars and buffers objects.');
 let bytes=0;
 for(const b of metadata.bindings){const spec=config.buffers[b.name];if(!spec||!Number.isInteger(spec.records)||spec.records<1||spec.records>1048576)throw Error(`${b.name}: records must be in [1,1048576].`);if(!['zero','one','ramp','random','sphere'].includes(spec.fill))throw Error(`${b.name}: unknown fill pattern.`);if(spec.fill==='sphere'&&b.elementType!=='vec4<f32>')throw Error('sphere fill requires float4 storage.');bytes+=spec.records*b.stride;}
 for(const b of metadata.bindings)for(const key of ['scale','offset'])if(config.buffers[b.name][key]!==undefined&&!Number.isFinite(config.buffers[b.name][key]))throw Error(`${b.name}: ${key} must be a finite number.`);
 if(bytes>64*1024*1024)throw Error('Sandbox buffers are limited to 64 MiB total.');
 if(!metadata.bindings.some(b=>b.name===config.output))throw Error('Choose an existing output buffer.');
 return bytes;
}
export function seedBuffer(binding,spec){
 const Type=binding.elementType.includes('u32')?Uint32Array:binding.elementType.includes('i32')?Int32Array:Float32Array;
 const data=new Type(spec.records*binding.stride/4);let state=314159;
 const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296;};
 for(let i=0;i<data.length;i++)data[i]=spec.fill==='one'?1:spec.fill==='ramp'?i%256:spec.fill==='random'?(Type===Float32Array?random()*2-1:Math.floor(random()*256)):0;
 if(spec.fill==='sphere')for(let i=0;i<spec.records;i++){const angle=random()*Math.PI*2,r=1+random()*6;data.set([Math.cos(angle)*r,(random()-0.5)*2,Math.sin(angle)*r,random()],i*4);}
 if(spec.scale!==undefined||spec.offset!==undefined)for(let i=0;i<data.length;i++){data[i]=data[i]*(spec.scale??1)+(spec.offset??0);if(!Number.isFinite(data[i]))throw Error('Buffer scale/offset exceeds the storage type range.');}
 return data;
}
