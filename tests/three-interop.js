/** Integration test: CUDA changes a shared position; actual rendered pixels must move. */
export async function runThreeInteropTest(runtime){
  const THREE=await import('three/webgpu'),{instanceIndex,float,color}=await import('three/tsl'),{createSharedFloat4}=await import('../src/runtime/three-bridge.js');
  const renderer=new THREE.WebGPURenderer({device:runtime.device,antialias:false});renderer.setSize(64,64);renderer.setClearColor(0x000000,1);await renderer.init();
  const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(-2,2,2,-2,0.1,10);camera.position.z=5;
  const shared=createSharedFloat4(renderer,runtime,new Float32Array([-1,0,0,1]));
  const material=new THREE.SpriteNodeMaterial();material.positionNode=shared.node.element(instanceIndex).xyz;material.scaleNode=float(0.5);material.colorNode=color(0xffffff);
  const sprite=new THREE.Sprite(material);sprite.count=1;sprite.frustumCulled=false;scene.add(sprite);
  const target=new THREE.RenderTarget(64,64,{type:THREE.UnsignedByteType,format:THREE.RGBAFormat});
  const pixel=async()=>{renderer.setRenderTarget(target);renderer.render(scene,camera);return await renderer.readRenderTargetPixelsAsync(target,0,0,64,64);};
  try{
    const before=await pixel();
    const kernel=await runtime.kernel('__global__ void move(float4* position){if(threadIdx.x==0u){float4 p=position[0];p.x=1.0f;position[0]=p;}}',{workgroupSize:[1,1,1]});
    runtime.batch().dispatch(kernel.bind({position:shared.resource}),[1]).submit();const after=await pixel();
    const bright=(pixels,x)=>pixels[(32*64+x)*4]>128;
    if(!bright(before,16)||bright(before,48)||bright(after,16)||!bright(after,48))throw new Error('Rendered pixels did not move from the left to the right after the CUDA kernel updated the shared GPUBuffer.');
    return {name:'Three r186 compute-to-render shared-buffer pixel test',pass:true,mode:'REAL WebGPU render readback',cpuUploadAfterCompute:false};
  }finally{renderer.setRenderTarget(null);await runtime.idle();material.dispose();shared.dispose();target.dispose();renderer.dispose();}
}
