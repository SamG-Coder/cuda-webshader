// Explicit ownership for object storage shared by multiple compiled kernels.
export class ObjectArena {
  constructor(runtime){this.runtime=runtime;this.buffers=[];this.signature=null;this.disposed=false;}
  bind(kernel){
    this.assertAlive();if(kernel.runtime!==this.runtime)throw Error('Object arena belongs to another runtime.');
    const types=kernel.artifact.metadata.objectHeap?.types;if(!kernel.objectLayout||!types)throw Error('Kernel does not use a persistent object arena.');
    const signature=JSON.stringify(types);if(this.signature&&this.signature!==signature)throw Error('Object arena layout or type tags do not match this kernel.');
    if(!this.signature){
      for(const t of types)if(t.byteLength>this.runtime.device.limits.maxStorageBufferBindingSize)throw Error('Object pool exceeds the storage binding limit.');
      try{for(const t of types)this.buffers.push(this.runtime.createBuffer(t.byteLength,{label:'Object pool '+t.name}));this.signature=signature;}catch(error){for(const b of this.buffers)this.runtime.destroyBuffer(b);this.buffers=[];throw error;}
    }
    const group=this.runtime.device.createBindGroup({layout:kernel.objectLayout,entries:types.map((t,i)=>({binding:t.binding,resource:{buffer:this.buffers[i].gpuBuffer}}))});this.runtime.stats.bindGroupsCreated++;return group;
  }
  assertAlive(){this.runtime.assertAlive();if(this.disposed)throw Error('Object arena is disposed.');for(const b of this.buffers)this.runtime.checkResource(b);}
  dispose(){if(this.disposed)return;for(const b of this.buffers)if(!b.destroyed)this.runtime.destroyBuffer(b);this.buffers=[];this.disposed=true;}
}
