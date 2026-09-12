export async function checkVolumeTransfer(runtime) {
 const source=await(await fetch('/tests/volume-transfer-kernel.cuh')).text();
 const kernel=await runtime.kernel(source,{entry:'d_integrate_trapezoidal',workgroupSize:[32,1,1]});
 const probeSource=await(await fetch('/tests/float4-surface-kernel.cuh')).text();
 const reader=await runtime.kernel(probeSource,{entry:'readTransfer',workgroupSize:[32,1,1]});
 const data=Float32Array.from({length:256},(_,i)=>{const x=i>>2;return [x*.125,-x*.25,(x%7)*.0625,1-x*.03125][i%4];});
 const texture=runtime.createTexture1D(data),comparisons=[];
 try {for(const size of [64,37,1024]){
  const width=Math.ceil(size/32)*32,surface=runtime.createTexture2D(new Float32Array(width*4).fill(-77),{width,height:1,format:'rgba32float',storage:true,addressMode:'clamp-to-edge'}),output=runtime.createBuffer(width*16);
  try{
   runtime.batch().dispatch(kernel.bind({transferTex:texture,transferIntegrateSurf:surface},{'extent.width':size,'extent.height':0,'extent.depth':0}),[width/32,1,1]).dispatch(reader.bind({input:surface,output},{count:width}),[width/32,1,1]).submit();
   const actual=await runtime.read(output),expected=new Float32Array(await(await fetch('/reports/volume-transfer'+(size===64?'':'-'+size)+'-native.bin')).arrayBuffer());
   if(expected.length!==size*4)throw Error('Invalid integration native capture');
   let maxError=0;for(let i=0;i<size*4;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite integration');maxError=Math.max(maxError,Math.abs(actual[i]-expected[i]));}
   for(let i=size*4;i<actual.length;i++)if(actual[i]!==-77)throw Error('Integration tail texel changed');
   if(maxError!==0)throw Error('Original transfer integration native mismatch: '+maxError);
   comparisons.push({size,components:size*4,maxError});
  }finally{runtime.destroyTexture(surface);runtime.destroyBuffer(output);}
 }return {comparisons,originalKernelUnchanged:true};}finally{runtime.destroyTexture(texture);}
}
