export async function checkFloat4Surface(runtime) {
 const source=await(await fetch('/tests/float4-surface-kernel.cuh')).text();
 const writer=await runtime.kernel(source,{entry:'writeTransfer',workgroupSize:[32,1,1]}),reader=await runtime.kernel(source,{entry:'readTransfer',workgroupSize:[32,1,1]}),zero=await runtime.kernel(source,{entry:'readTransferZero',workgroupSize:[1,1,1]});
 const data=Float32Array.from({length:256},(_,i)=>{const x=i>>2;return [x*.125,-x*.25,(x%7)*.0625,1-x*.03125][i%4];});
 const input=runtime.createBuffer(data),output=runtime.createBuffer(new Float32Array(272).fill(-12345));
 const surface=runtime.createTexture2D(null,{width:64,height:1,format:'rgba32float',storage:true,filter:'linear',addressMode:'clamp-to-edge'});
 const wrong=runtime.createTexture2D(null,{width:64,height:2,format:'rgba32float',storage:true});
 try {
  const write=writer.bind({input,output:surface},{count:64}),before={...runtime.stats};
  runtime.batch().dispatch(write,[2,1,1]).dispatch(reader.bind({input:surface,output},{count:64}),[2,1,1]).submit();await runtime.idle();
  if(runtime.stats.readbackBytes!==before.readbackBytes||runtime.stats.dataBytesUploaded!==before.dataBytesUploaded)throw Error('Intermediate surface transferred through CPU');
  const actual=await runtime.read(output),expected=new Float32Array(await(await fetch('/reports/float4-surface-native.bin')).arrayBuffer());
  if(expected.length!==256)throw Error('Invalid native capture');
  for(let i=0;i<actual.length;i++)if(actual[i]!== (i<256?expected[i]:-12345))throw Error('Float4 surface/native mismatch at '+i);
  runtime.batch().dispatch(zero.bind({input:surface,output}),[1,1,1]).submit();const first=await runtime.read(output);
  for(let i=0;i<4;i++)if(first[i]!==data[i])throw Error('Integer zero texture coordinate mismatch');
  for(const [invocation,groups] of [[write,[3,1,1]],[write,[2,2,1]],[write,[2,1,2]],[writer.bind({input,output:wrong},{count:64}),[2,1,1]]]){
   const batch=runtime.batch();let rejected=false;try{batch.dispatch(invocation,groups);}catch(e){if(!/Surface dispatch/.test(e.message))throw e;rejected=true;}finally{batch.discard();}if(!rejected)throw Error('Unsafe surface extent accepted');
  }
  // Tail lanes must leave untouched texels intact, even when the allocation
  // covers a complete workgroup and the logical input count does not.
  const initial=new Float32Array(256).fill(-77),partial=runtime.createTexture2D(initial,{width:64,height:1,format:'rgba32float',storage:true,addressMode:'clamp-to-edge'});
  try{runtime.batch().dispatch(writer.bind({input,output:partial},{count:37}),[2,1,1]).dispatch(reader.bind({input:partial,output},{count:64}),[2,1,1]).submit();const tail=await runtime.read(output);for(let i=0;i<256;i++)if(tail[i]!== (i<148?data[i]:-77))throw Error('Partial surface tail changed');}finally{runtime.destroyTexture(partial);}
  return {nativeFloatComponents:256,exact:true,partialCount:37,integerCoordinate:true,unsafeExtentsRejected:4,noIntermediateCPUTransfer:true};
 } finally {runtime.destroyBuffer(input);runtime.destroyBuffer(output);runtime.destroyTexture(surface);runtime.destroyTexture(wrong);}
}
