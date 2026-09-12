/** WebGPU runtime: cached pipelines/bindings, batched dispatch and a per-batch uniform snapshot arena. */
import {compile} from '../compiler/compiler.js';
const roundUp = (n, alignment) => Math.ceil(n / alignment) * alignment;
let resourceId = 0;
export function packScalars(metadata, values, target = new ArrayBuffer(metadata.uniformSize)) {
  if (target.byteLength < metadata.uniformSize) throw new RangeError('Uniform destination is too small.');
  const known = new Set(metadata.scalars.map(s => s.name));
  for (const name of Object.keys(values)) if (!known.has(name)) throw new Error(`Unknown scalar parameter '${name}'.`);
  const writes = [];
  for (const p of metadata.scalars) {
    const v = Object.hasOwn(values,p.name)?values[p.name]:p.origin==='constant'?p.defaultValue:undefined;
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(`Scalar '${p.name}' must be finite.`);
    if (p.type === 'u32') { if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) throw new RangeError(`${p.name} is not a u32.`); }
    else if (p.type === 'i32') { if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) throw new RangeError(`${p.name} is not an i32.`); }
    else if (!Number.isFinite(Math.fround(v))) throw new RangeError(`${p.name} overflows f32.`);
    writes.push([p,v]);
  }
  // Validate the complete update before touching the destination: failed updates are transactional.
  const view = new DataView(target);
  for(const [p,v] of writes) {
    if(p.type === 'u32') view.setUint32(p.offset,v,true);
    else if(p.type === 'i32') view.setInt32(p.offset,v,true);
    else view.setFloat32(p.offset,v,true);
  }
  return target;
}
export function validateWorkgroup(metadata, limits) {
  const size = metadata.workgroupSize;
  for (let i = 0; i < 3; i++) if (size[i] > limits[['maxComputeWorkgroupSizeX','maxComputeWorkgroupSizeY','maxComputeWorkgroupSizeZ'][i]]) throw new Error(`Workgroup ${size} exceeds a device dimension limit.`);
  if (size.reduce((a,b)=>a*b,1)>limits.maxComputeInvocationsPerWorkgroup) throw new Error(`Workgroup ${size} exceeds maxComputeInvocationsPerWorkgroup.`);
  if (metadata.workgroupStorageBytes > limits.maxComputeWorkgroupStorageSize) throw new Error(`Kernel requires ${metadata.workgroupStorageBytes} workgroup bytes; device supports ${limits.maxComputeWorkgroupStorageSize}.`);
  if (metadata.uniformSize > limits.maxUniformBufferBindingSize) throw new Error('Uniform parameters exceed the device binding limit.');
  if (metadata.bindings.length > limits.maxStorageBuffersPerShaderStage) throw new Error('Too many storage buffers for this device.');
}
export class GpuRuntime {
  static async create(options = {}) {
    if (!globalThis.navigator?.gpu && !options.device) throw new Error('WebGPU is required. Open this project on localhost or HTTPS in a WebGPU-capable browser. WebGL cannot run these kernels.');
    const adapter = options.adapter || (!options.device ? await navigator.gpu.requestAdapter({powerPreference:'high-performance'}) : null);
    if (!adapter && !options.device) throw new Error('No WebGPU adapter is available. Check the browser GPU settings and graphics driver.');
    const features = ['timestamp-query','core-features-and-limits'].filter(f => adapter?.features.has(f));
    const requiredLimits = adapter ? {
      maxComputeInvocationsPerWorkgroup: Math.min(adapter.limits.maxComputeInvocationsPerWorkgroup, 1024),
      maxComputeWorkgroupSizeX: Math.min(adapter.limits.maxComputeWorkgroupSizeX,1024),
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 256 * 1024 * 1024)
    } : undefined;
    const device = options.device || await adapter.requestDevice({requiredFeatures:features, requiredLimits});
    return new GpuRuntime(device, {...options,adapter,ownsDevice:!options.device});
  }
  constructor(device, options = {}) {
    this.device=device; this.adapter=options.adapter; this.ownsDevice=options.ownsDevice ?? false;
    this.disposed=false; this.lost=null; this.onError=options.onError || (error=>console.error(error));
    this.buffers=new Set(); this.pipelineCache=new Map(); this.pipelineQueue=Promise.resolve();
    this.uniformAlignment=device.limits.minUniformBufferOffsetAlignment;
    this.uniformCapacity=roundUp(options.uniformCapacity || 65536,this.uniformAlignment);
    this.uniformBuffer=device.createBuffer({label:'CUDA WebShader uniform snapshot arena',size:this.uniformCapacity,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.batchMemory=[];
    this.stats={pipelineCompiles:0,pipelineCacheHits:0,bindGroupsCreated:0,submissions:0,dispatches:0,uniformBytesUploaded:0,dataBytesUploaded:0,readbackBytes:0};
    this.errorListener=e=>this.onError(e.error || e);
    device.addEventListener('uncapturederror',this.errorListener);
    device.lost.then(info=>{this.lost=info;if(info.reason!=='destroyed'&&!this.disposed)this.onError(new Error(`GPU device lost: ${info.message}. Reload to recreate GPU resources.`));});
  }
  assertAlive() { if(this.disposed)throw new Error('Runtime is disposed.');if(this.lost)throw new Error(`GPU device lost: ${this.lost.message}`); }
  describe() {
    const info=this.adapter?.info;
    return {vendor:info?.vendor || 'not exposed',architecture:info?.architecture || '',device:info?.device || '',description:info?.description || '',features:[...this.device.features],timestampQuery:this.device.features.has('timestamp-query'),limits:{maxComputeInvocationsPerWorkgroup:this.device.limits.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupStorageSize:this.device.limits.maxComputeWorkgroupStorageSize,maxStorageBufferBindingSize:this.device.limits.maxStorageBufferBindingSize}};
  }
  createBuffer(dataOrBytes, {label='compute buffer',usage=0} = {}) {
    this.assertAlive(); const data=ArrayBuffer.isView(dataOrBytes)?dataOrBytes:null, bytes=data?data.byteLength:dataOrBytes;
    if(!Number.isSafeInteger(bytes)||bytes<0)throw new RangeError('Buffer size must be a nonnegative integer.');
    const size=Math.max(4,roundUp(bytes,4));
    if(size>this.device.limits.maxStorageBufferBindingSize)throw new RangeError('Buffer exceeds maxStorageBufferBindingSize.');
    const gpuBuffer=this.device.createBuffer({label,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|usage,mappedAtCreation:!!data});
    if(data){new Uint8Array(gpuBuffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));gpuBuffer.unmap();this.stats.dataBytesUploaded+=bytes;}
    const r={id:++resourceId,gpuBuffer,byteLength:bytes,size,label,runtime:this,owned:true,destroyed:false};this.buffers.add(r);return r;
  }
  importBuffer(gpuBuffer, byteLength=gpuBuffer.size, label='imported storage buffer') {
    this.assertAlive();
    if(!(gpuBuffer.usage&GPUBufferUsage.STORAGE))throw new Error('Imported GPUBuffer must have STORAGE usage.');
    if(byteLength<0||byteLength>gpuBuffer.size)throw new RangeError('Invalid imported buffer length.');
    return {id:++resourceId,gpuBuffer,byteLength,size:gpuBuffer.size,label,runtime:this,owned:false,destroyed:false};
  }
  checkResource(resource) { if(!resource||resource.runtime!==this||resource.destroyed)throw new Error('Buffer is destroyed or belongs to a different runtime.'); }
  destroyBuffer(resource) { this.checkResource(resource); if(resource.owned){resource.gpuBuffer.destroy();this.buffers.delete(resource);}resource.destroyed=true; }
  write(resource,data,offset=0) {
    this.assertAlive();this.checkResource(resource);
    if(!ArrayBuffer.isView(data)||offset%4||data.byteLength%4||offset<0||offset+data.byteLength>resource.size)throw new RangeError('Write must be aligned and fit in the destination.');
    this.device.queue.writeBuffer(resource.gpuBuffer,offset,data.buffer,data.byteOffset,data.byteLength);this.stats.dataBytesUploaded+=data.byteLength;
  }
  async read(resource, Type=Float32Array, byteLength=resource.byteLength, offset=0) {
    this.assertAlive();this.checkResource(resource);
    if(![Float32Array,Uint32Array,Int32Array].includes(Type))throw new TypeError('Readback supports 32-bit float/integer arrays.');
    if(!Number.isInteger(byteLength)||byteLength<0||byteLength%4||offset%4||offset<0||offset+byteLength>resource.size)throw new RangeError('Invalid readback range.');
    if(byteLength===0)return new Type(0);
    const staging=this.device.createBuffer({label:'explicit readback (not render path)',size:byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(resource.gpuBuffer,offset,staging,0,byteLength);this.device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);const result=new Type(staging.getMappedRange().slice(0));staging.unmap();this.stats.readbackBytes+=byteLength;return result;}finally{staging.destroy();}
  }
  async kernel(sourceOrArtifact,options={}) {
    this.assertAlive();const artifact=typeof sourceOrArtifact==='string'?compile(sourceOrArtifact,options):sourceOrArtifact;
    validateWorkgroup(artifact.metadata,this.device.limits);
    // Use complete source/ABI, not an unchecked short hash, as the cache key.
    const key=artifact.wgsl+'\n'+JSON.stringify(artifact.metadata);
    if(this.pipelineCache.has(key)){this.stats.pipelineCacheHits++;return this.pipelineCache.get(key);}
    const build=async()=>{
      this.device.pushErrorScope('validation');
      let thrown=null,result;
      try{
        const module=this.device.createShaderModule({label:`CUDA -> WGSL: ${artifact.name}`,code:artifact.wgsl});
        const info=await module.getCompilationInfo();
        const errors=info.messages.filter(m=>m.type==='error');
        if(errors.length)throw new Error(`${artifact.name}: WGSL validation failed\n`+errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
        const entries=artifact.metadata.bindings.map(b=>({binding:b.binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:b.readOnly?'read-only-storage':'storage',minBindingSize:b.stride}}));
        if(artifact.metadata.uniformSize)entries.push({binding:artifact.metadata.uniformBinding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:artifact.metadata.uniformSize}});
        const layout=this.device.createBindGroupLayout({label:artifact.name,entries});
        const pipeline=await this.device.createComputePipelineAsync({label:artifact.name,layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:artifact.entryPoint || 'main'}});
        this.stats.pipelineCompiles++; result=new Kernel(this,artifact,pipeline,layout,info.messages);
      }catch(error){thrown=error;}
      const validation=await this.device.popErrorScope();
      if(thrown)throw thrown;if(validation)throw new Error(validation.message);return result;
    };
    // Error scopes are stack-based. Serialize pipeline creation to prevent interleaved scopes.
    const promise=this.pipelineQueue.then(build);this.pipelineQueue=promise.catch(()=>{});this.pipelineCache.set(key,promise);
    promise.catch(()=>this.pipelineCache.delete(key));return promise;
  }
  batch(options={}) {this.assertAlive();return new ComputeBatch(this,options);}
  async idle() {this.assertAlive();await this.device.queue.onSubmittedWorkDone();}
  dispose() {if(this.disposed)return;this.disposed=true;for(const b of this.buffers){b.gpuBuffer.destroy();b.destroyed=true;}this.buffers.clear();this.uniformBuffer.destroy();this.pipelineCache.clear();this.batchMemory=[];this.device.removeEventListener('uncapturederror',this.errorListener);if(this.ownsDevice)this.device.destroy();}
}
export class Kernel {
  constructor(runtime,artifact,pipeline,layout,messages){this.runtime=runtime;this.artifact=artifact;this.pipeline=pipeline;this.layout=layout;this.messages=messages;}
  bind(buffers,scalars={}) {return new Invocation(this,buffers,scalars);}
}
export class Invocation {
  constructor(kernel,buffers,scalars) {
    this.kernel=kernel;this.runtime=kernel.runtime;this.runtime.assertAlive();this.version=0;this.values={};
    this.uniformData=new ArrayBuffer(kernel.artifact.metadata.uniformSize);this.buffers={...buffers};
    const meta=kernel.artifact.metadata,entries=[],seen=new Map(),known=new Set(meta.bindings.map(b=>b.name));
    for(const name of Object.keys(buffers))if(!known.has(name))throw new Error(`Unknown buffer '${name}'.`);
    for(const b of meta.bindings){
      const resource=buffers[b.name];this.runtime.checkResource(resource);
      if(resource.size<b.stride)throw new RangeError(`Buffer ${b.name} is smaller than one ${b.elementType} record.`);
      if(resource.size%b.stride)throw new RangeError(`Buffer ${b.name} is not aligned to ${b.stride}-byte records.`);
      if(seen.has(resource.gpuBuffer)&&(!b.readOnly||!seen.get(resource.gpuBuffer)))throw new Error('Writable buffer aliasing across bindings is rejected; use separate buffers or a single in-place parameter.');
      seen.set(resource.gpuBuffer,b.readOnly);entries.push({binding:b.binding,resource:{buffer:resource.gpuBuffer,offset:0,size:resource.size}});
    }
    if(meta.uniformSize)entries.push({binding:meta.uniformBinding,resource:{buffer:this.runtime.uniformBuffer,offset:0,size:meta.uniformSize}});
    this.bindGroup=this.runtime.device.createBindGroup({label:`${kernel.artifact.name}: persistent bindings`,layout:kernel.layout,entries});this.runtime.stats.bindGroupsCreated++;
    this.setScalars(scalars);
  }
  setScalars(values) {const merged={...this.values,...values};packScalars(this.kernel.artifact.metadata,merged,this.uniformData);this.values=merged;this.version++;return this;}
}
export class ComputeBatch {
  constructor(runtime,{label='compute batch',timestampWrites}={}) {
    this.runtime=runtime;this.encoder=runtime.device.createCommandEncoder({label});this.pass=null;this.ended=false;
    this.data=runtime.batchMemory.pop() || new Uint8Array(runtime.uniformCapacity);this.cursor=0;this.snapshots=new Map();
    this.timestampWrites=timestampWrites;this.passCount=0;this.lastPipeline=null;this.lastBindGroup=null;this.lastOffset=-1;this.dispatchCount=0;
  }
  assertOpen(){if(this.ended)throw new Error('Batch has already been submitted or discarded.');this.runtime.assertAlive();}
  beginPass(){if(!this.pass){if(this.timestampWrites&&this.passCount)throw new Error('Timestamped batch supports a single compute pass.');this.pass=this.encoder.beginComputePass(this.timestampWrites?{timestampWrites:this.timestampWrites}:{});this.passCount++;this.lastPipeline=null;this.lastBindGroup=null;this.lastOffset=-1;}return this.pass;}
  endPass(){if(this.pass){this.pass.end();this.pass=null;}}
  dispatch(invocation,workgroups) {
    this.assertOpen();if(invocation.runtime!==this.runtime)throw new Error('Invocation belongs to another runtime.');
    const groups=Array.isArray(workgroups)?[...workgroups]:[workgroups];while(groups.length<3)groups.push(1);
    if(groups.length!==3||groups.some(x=>!Number.isSafeInteger(x)||x<0||x>this.runtime.device.limits.maxComputeWorkgroupsPerDimension))throw new RangeError('Invalid workgroup counts. These are block counts, not thread counts.');
    if(groups.some(x=>x===0))return this;
    for(const r of Object.values(invocation.buffers))this.runtime.checkResource(r);
    const meta=invocation.kernel.artifact.metadata;let offset=0;
    if(meta.uniformSize){
      const previous=this.snapshots.get(invocation);
      if(previous?.version===invocation.version)offset=previous.offset;
      else{
        offset=roundUp(this.cursor,this.runtime.uniformAlignment);
        if(offset+meta.uniformSize>this.data.byteLength)throw new RangeError('Batch uniform arena is full. Submit smaller batches or raise uniformCapacity.');
        this.data.set(new Uint8Array(invocation.uniformData),offset);this.cursor=offset+meta.uniformSize;this.snapshots.set(invocation,{version:invocation.version,offset});
      }
    }
    const pass=this.beginPass();
    if(this.lastPipeline!==invocation.kernel.pipeline){pass.setPipeline(invocation.kernel.pipeline);this.lastPipeline=invocation.kernel.pipeline;}
    if(this.lastBindGroup!==invocation.bindGroup||this.lastOffset!==offset){pass.setBindGroup(0,invocation.bindGroup,meta.uniformSize?[offset]:[]);this.lastBindGroup=invocation.bindGroup;this.lastOffset=offset;}
    pass.dispatchWorkgroups(...groups);this.dispatchCount++;return this;
  }
  clear(resource){this.assertOpen();this.runtime.checkResource(resource);this.endPass();this.encoder.clearBuffer(resource.gpuBuffer,0,resource.size);return this;}
  submit(){
    this.assertOpen();this.endPass();this.ended=true;
    if(this.cursor){this.runtime.device.queue.writeBuffer(this.runtime.uniformBuffer,0,this.data,0,this.cursor);this.runtime.stats.uniformBytesUploaded+=this.cursor;}
    this.runtime.device.queue.submit([this.encoder.finish()]);this.runtime.stats.submissions++;this.runtime.stats.dispatches+=this.dispatchCount;
    this.runtime.batchMemory.push(this.data);this.data=null;this.snapshots.clear();
  }
  discard(){if(this.ended)return;this.endPass();this.ended=true;this.runtime.batchMemory.push(this.data);this.data=null;this.snapshots.clear();}
}
