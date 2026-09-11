import * as THREE from 'three/webgpu';
import {storage} from 'three/tsl';
/**
 * The ONLY module coupled to Three r186 backend internals.
 * Three owns the allocation; the compute runtime borrows the exact GPUBuffer on the same GPUDevice.
 * Never set attribute.needsUpdate after compute writes: that would upload the stale CPU seed array.
 */
export function createSharedFloat4(renderer,runtime,array,{label='shared float4 records'}={}){
  if(THREE.REVISION!=='186')throw new Error(`Interop is pinned to Three r186, found r${THREE.REVISION}. Re-run interop tests before upgrading.`);
  if(!renderer.backend?.isWebGPUBackend||renderer.backend.device!==runtime.device)throw new Error('Compute and Three.js must use the exact same initialized WebGPU device.');
  if(!(array instanceof Float32Array)||array.length===0||array.length%4)throw new TypeError('Provide a nonempty Float32Array of float4 records.');
  const attribute=new THREE.StorageInstancedBufferAttribute(array,4);attribute.name=label;
  renderer.backend.createStorageAttribute(attribute);
  const gpuBuffer=renderer.backend.get(attribute).buffer;
  if(!gpuBuffer)throw new Error('Three.js did not allocate a storage GPUBuffer.');
  const resource=runtime.importBuffer(gpuBuffer,array.byteLength,label),node=storage(attribute,'vec4',array.length/4).toReadOnly();
  let disposed=false;
  return {attribute,node,resource,gpuBuffer,dispose(){if(disposed)return;disposed=true;runtime.destroyBuffer(resource);renderer.backend.destroyAttribute(attribute);}};
}
