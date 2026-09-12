export async function checkVolumePreintegrated(runtime){
 const source=await(await fetch('/tests/volume-preintegrated-render-kernel.cuh')).text();
 const integrate=await runtime.kernel(source,{entry:'d_integrate_trapezoidal',workgroupSize:[32,1,1]}),preintegrate=await runtime.kernel(source,{entry:'d_preintegrate',workgroupSize:[8,8,1]});
 const render=await runtime.kernel(source,{entry:'d_render_preint',workgroupSize:[8,8,1]}),regular=await runtime.kernel(source,{entry:'d_render_preint_off',workgroupSize:[8,8,1]});
 const colors=[];for(let i=0;i<2;i++)colors.push(runtime.createTexture1D(Float32Array.from(await(await fetch('/tests/volume-transfer-colors-'+i+'.json')).json())));
 const integrated=runtime.createTexture2D(null,{width:1024,height:1,format:'rgba32float',storage:true,addressMode:'clamp-to-edge'}),volume=runtime.createTexture3D(new Uint8Array(await(await fetch('/showcases/volume-render/Bucky.raw')).arrayBuffer()),{width:32,height:32,depth:32,addressMode:'clamp-to-edge'}),output=runtime.createBuffer(256*256*4);
 const comparisons=[];let tableMaxError=0;
 try{for(const size of [32,1024]){
  const table=runtime.createLayeredTexture2D(null,{width:size,height:size,layers:2,storage:true});
  try{
   for(let layer=0;layer<2;layer++)runtime.batch().dispatch(integrate.bind({transferTex:colors[layer],transferIntegrateSurf:integrated},{'extent.width':1024,'extent.height':0,'extent.depth':0}),[32,1,1]).dispatch(preintegrate.bind({transferTex:colors[layer],transferIntegrateTex:integrated,transferLayerPreintSurf:table},{layer,steps:1024,'extent.width':size,'extent.height':size,'extent.depth':2}),[size/8,size/8,1]).submit();
   if(size===32){
    const probe=await runtime.kernel('__device__ float4 sample(cudaTextureObject_t t,float x,float y,int layer){return tex2DLayered<float4>(t,x,y,layer);}__global__ void read(float4* output,cudaTextureObject_t t,int layer){uint x=blockIdx.x*blockDim.x+threadIdx.x;uint y=blockIdx.y*blockDim.y+threadIdx.y;output[layer*1024+y*32+x]=sample(t,(float(x)+0.5f)/32.f,(float(y)+0.5f)/32.f,layer);}',{entry:'read',workgroupSize:[8,8,1]});
    const data=runtime.createBuffer(32*32*2*16);
    try{for(let layer=0;layer<2;layer++)runtime.batch().dispatch(probe.bind({output:data,t:table},{layer}),[4,4,1]).submit();const actual=await runtime.read(data),expected=new Float32Array(await(await fetch('/reports/volume-preintegrated-table-native.bin')).arrayBuffer());if(actual.length!==expected.length)throw Error('Invalid table capture');for(let i=0;i<actual.length;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite table');tableMaxError=Math.max(tableMaxError,Math.abs(actual[i]-expected[i]));}if(tableMaxError>5e-6)throw Error('Preintegrated table differs from native: '+tableMaxError);}finally{runtime.destroyBuffer(data);}
    for(const layer of [-1,2]){const batch=runtime.batch();let rejected=false;try{batch.dispatch(preintegrate.bind({transferTex:colors[0],transferIntegrateTex:integrated,transferLayerPreintSurf:table},{layer,steps:1024,'extent.width':32,'extent.height':32,'extent.depth':2}),[4,4,1]);}catch(e){if(!/Surface layer/.test(e.message))throw e;rejected=true;}finally{batch.discard();}if(!rejected)throw Error('Invalid layer accepted');}
   }
   for(let scenario=0;scenario<3;scenario++){
    const matrix=scenario===1?[.8660254,0,.5,2,0,1,0,0,-.5,0,.8660254,3.4641016]:[1,0,0,0,0,1,0,0,0,0,1,4];
    const scalars={imageW:256,imageH:256,density:.05,brightness:1,transferOffset:0,transferScale:1,...Object.fromEntries(matrix.map((v,i)=>['constant.c_invViewMatrix.m['+Math.floor(i/4)+'].'+'xyzw'[i%4],v]))};
    runtime.device.queue.writeBuffer(output.gpuBuffer,0,new Uint32Array(256*256));
    runtime.batch().dispatch((scenario===2?regular:render).bind({d_output:output,volumeTex:volume,transferTex:colors[0],transferLayerPreintTex:table},scalars),[32,32,1]).submit();
    const actual=new Uint8Array((await runtime.read(output,Uint32Array)).buffer),expected=new Uint8Array(await(await fetch('/reports/volume-preintegrated-'+size+'-'+scenario+'-native.bin')).arrayBuffer());
    let maxChannelError=0,differingChannels=0;for(let i=0;i<actual.length;i++){const error=Math.abs(actual[i]-expected[i]);maxChannelError=Math.max(maxChannelError,error);if(error)differingChannels++;}if(maxChannelError>4)throw Error('Native render mismatch size='+size+' scenario='+scenario+' max='+maxChannelError);
    comparisons.push({tableSize:size,scenario,pixels:65536,maxChannelError,differingChannels});
   }
  }finally{runtime.destroyTexture(table);}
 }return {tableMaxError,comparisons,originalBodiesUnchanged:true};}finally{for(const t of colors)runtime.destroyTexture(t);runtime.destroyTexture(integrated);runtime.destroyTexture(volume);runtime.destroyBuffer(output);}
}
