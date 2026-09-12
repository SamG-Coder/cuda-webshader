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
 if(config.passes!==undefined&&(!Array.isArray(config.passes)||config.passes.length>8||config.passes.some(p=>!p||typeof p.entry!=='string'||!Array.isArray(p.block)||p.block.length!==3)))throw Error('Additional passes require entry, three block dimensions and groups; at most eight passes are supported.');
 if(!Array.isArray(config.groups)||config.groups.length!==3||config.groups.some(n=>!Number.isInteger(n)||n<1||n>65535)||config.groups.reduce((a,b)=>a*b,1)*metadata.workgroupSize.reduce((a,b)=>a*b,1)>4194304)throw Error('Launch must contain three positive block counts, each ≤65,535, and at most 4,194,304 invocations.');
 if(!config.scalars||!config.buffers)throw Error('Settings need scalars and buffers objects.');
 validateCopies(config.copies,metadata,config);
 for(const pass of config.passes||[])validateCopies(pass.copies,metadata,config);
 let bytes=0;
 for(const name of Object.keys(config.textures||{}))if(!metadata.textures?.some(t=>t.name===name))throw Error('Unknown texture '+name);
 for(const t of metadata.textures||[]){const spec=config.textures?.[t.name];if(t.format==='rgba32float'){if(!spec||!Array.isArray(spec.dimensions)||spec.dimensions.length!==1||!Number.isInteger(spec.dimensions[0])||spec.dimensions[0]<1||!Array.isArray(spec.values)||spec.values.length!==spec.dimensions[0]*4||spec.values.some(v=>typeof v!=='number'||!Number.isFinite(Math.fround(v)))||!['linear','nearest'].includes(spec.filter)||!['repeat','clamp-to-edge','mirror-repeat'].includes(spec.addressMode))throw Error('Float4 transfer texture requires a width, four finite values per record, filter and addressMode.');bytes+=spec.values.length*4;continue;}if(!spec||!Array.isArray(spec.dimensions)||spec.dimensions.length!==3||spec.dimensions.some(n=>!Number.isInteger(n)||n<1||n>2048)||typeof spec.source!=='string'||!['linear','nearest'].includes(spec.filter)||!['repeat','clamp-to-edge','mirror-repeat'].includes(spec.addressMode))throw Error('Texture requires dimensions, a byte-data source, filter and addressMode.');bytes+=spec.dimensions.reduce((a,b)=>a*b,1);}

 for(const b of metadata.bindings){const spec=config.buffers[b.name];if(!spec||!Number.isInteger(spec.records)||spec.records<1||spec.records>1048576)throw Error(`${b.name}: records must be in [1,1048576].`);if(!['zero','one','ramp','random','sphere','rgba'].includes(spec.fill))throw Error(`${b.name}: unknown fill pattern.`);if(spec.fill==='rgba'&&(b.elementType!=='u32'||!Number.isInteger(spec.width)||spec.width<1||spec.records%spec.width))throw Error('rgba fill requires uint storage and a positive width dividing records.');if(spec.fill==='sphere'&&b.elementType!=='vec4<f32>')throw Error('sphere fill requires float4 storage.');bytes+=spec.records*b.stride;}
 for(const b of metadata.bindings)for(const key of ['scale','offset'])if(config.buffers[b.name][key]!==undefined&&!Number.isFinite(config.buffers[b.name][key]))throw Error(`${b.name}: ${key} must be a finite number.`);
 if(bytes>64*1024*1024)throw Error('Sandbox buffers are limited to 64 MiB total.');
 if(!metadata.bindings.some(b=>b.name===config.output))throw Error('Choose an existing output buffer.');
 if(config.image!==undefined){const v=config.image,b=metadata.bindings.find(b=>b.name===config.output);if(b.elementType!=='u32'||!['rgba8','r8-in-u32'].includes(v?.format)||![v.width,v.height].every(n=>Number.isInteger(n)&&n>0)||v.width*v.height!==config.buffers[config.output].records)throw Error('Image preview requires uint RGBA8 or r8-in-u32 storage and matching width/height.');}
 if(config.volume!==undefined){const v=config.volume,b=metadata.bindings.find(b=>b.name===config.output);if(b.elementType!=='f32'||!Array.isArray(v?.dimensions)||v.dimensions.length!==3||v.dimensions.some(n=>!Number.isInteger(n)||n<1)||!Number.isInteger(v.halo)||v.halo<0||v.dimensions.map(n=>n+2*v.halo).reduce((a,b)=>a*b,1)!==config.buffers[config.output].records)throw Error('Volume preview requires float storage, three positive interior dimensions, and a nonnegative halo matching the record count.');}
 if(config.feedback!==undefined){if(!config.feedback||typeof config.feedback!=='object'||Array.isArray(config.feedback))throw Error('Feedback must map destination buffer names to source names.');for(const [target,source]of Object.entries(config.feedback)){const a=metadata.bindings.find(b=>b.name===source),b=metadata.bindings.find(b=>b.name===target);if(!a||!b||source===target||a.elementType!==b.elementType||config.buffers[source].records!==config.buffers[target].records)throw Error('Feedback requires distinct buffers with matching types and record counts.');if(Object.hasOwn(config.feedback,source))throw Error('Feedback chains and cycles are unsupported.');}}
 return bytes;
}
export function preparePass(pass,metadata,config,rootMetadata){
 if(metadata.textures?.length)throw Error('Texture bindings in additional passes are not yet supported.');
 for(const name of Object.keys(pass.bindings||{}))if(!metadata.bindings.some(b=>b.name===name))throw Error(`Pass ${pass.entry}: unknown buffer parameter ${name}.`);
 for(const name of Object.keys(pass.scalars||{}))if(!metadata.scalars.some(s=>s.name===name))throw Error(`Pass ${pass.entry}: unknown scalar parameter ${name}.`);
 const bindings={},buffers={};for(const b of metadata.bindings){const name=pass.bindings?.[b.name]??b.name,root=rootMetadata.bindings.find(r=>r.name===name);if(!root||root.elementType!==b.elementType)throw Error(`Pass ${pass.entry}: ${b.name} needs a matching existing buffer.`);bindings[b.name]=name;buffers[b.name]=config.buffers[name];}
 const supplied={...config.scalars,...pass.scalars},scalars=Object.fromEntries(metadata.scalars.filter(s=>Object.hasOwn(supplied,s.name)).map(s=>[s.name,supplied[s.name]]));
 validateConfig({groups:pass.groups,scalars,buffers,output:metadata.bindings[0]?.name},metadata);
 return {bindings,scalars,groups:pass.groups};
}
export function seedBuffer(binding,spec){
 const Type=binding.elementType.includes('u32')?Uint32Array:binding.elementType.includes('i32')?Int32Array:Float32Array;
 const data=new Type(spec.records*binding.stride/4);let state=314159;
 const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296;};
 for(let i=0;i<data.length;i++)data[i]=spec.fill==='one'?1:spec.fill==='ramp'?i%256:spec.fill==='random'?(Type===Float32Array?random()*2-1:Math.floor(random()*256)):0;
 if(spec.fill==='rgba')for(let i=0;i<spec.records;i++){const x=i%spec.width,y=Math.floor(i/spec.width),h=spec.records/spec.width,r=Math.round(x/Math.max(1,spec.width-1)*255),g=Math.round(y/Math.max(1,h-1)*255),b=(Math.floor(x/16)+Math.floor(y/16))%2?240:20;data[i]=(r|(g<<8)|(b<<16)|(255<<24))>>>0;}
 if(spec.fill==='sphere')for(let i=0;i<spec.records;i++){const angle=random()*Math.PI*2,r=1+random()*6;data.set([Math.cos(angle)*r,(random()-0.5)*2,Math.sin(angle)*r,random()],i*4);}
 if(spec.scale!==undefined||spec.offset!==undefined)for(let i=0;i<data.length;i++){data[i]=data[i]*(spec.scale??1)+(spec.offset??0);if(!Number.isFinite(data[i]))throw Error('Buffer scale/offset exceeds the storage type range.');}
 return data;
}

export function validateCopies(copies,metadata,config){
 if(copies===undefined)return;
 if(!Array.isArray(copies)||copies.length>8)throw Error('Copies must be an array of at most eight GPU ranges.');
 for(const copy of copies){const a=metadata.bindings.find(b=>b.name===copy?.source),b=metadata.bindings.find(b=>b.name===copy?.target);if(!a||!b||a.name===b.name||a.elementType!==b.elementType)throw Error('Copy requires distinct buffers with matching element types.');const {sourceOffset=0,targetOffset=0,byteLength}=copy;if([sourceOffset,targetOffset,byteLength].some(n=>!Number.isSafeInteger(n)||n<0||n%4)||sourceOffset+byteLength>(config.buffers[a.name]?.records??0)*a.stride||targetOffset+byteLength>(config.buffers[b.name]?.records??0)*b.stride)throw Error('Copy range must be aligned and within both buffers.');}
}
