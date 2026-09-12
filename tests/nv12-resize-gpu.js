export async function checkNv12Resize(runtime){
 const source=await(await fetch('/tests/nv12-resize-kernel.cuh')).text();
 const kernel=await runtime.kernel(source,{entry:'resizeNV12BatchKernel',workgroupSize:[32,32,1]});
 const input=new Uint8Array(await(await fetch('/reports/nv12-input.bin')).arrayBuffer()),expected=new Uint8Array(await(await fetch('/reports/nv12-resize-stage-native.bin')).arrayBuffer());
 const luma=runtime.createByteTexture2D(input,{width:1920,height:1620}),chroma=runtime.createByteTexture2D(input,{width:960,height:1620,components:2});
 const output=runtime.createBuffer(new Uint8Array(640*720).fill(0xa5));
 try{
  const call=kernel.bind({texSrcLuma:luma,texSrcChroma:chroma,pDstNv12:output},{nSrcWidth:1920,nSrcHeight:1080,nDstPitch:640,nDstWidth:640,nDstHeight:480,nBatchSize:1});
  for(let repeat=0;repeat<3;repeat++){
   runtime.batch().dispatch(call,[11,8,1]).submit();
   const words=await runtime.read(output,Uint32Array),actual=new Uint8Array(words.buffer,words.byteOffset,words.byteLength);
   if(actual.length!==expected.length||actual.some((v,i)=>v!==expected[i]))throw Error('NV12 resize native byte mismatch');
  }
  return {width:1920,height:1080,dstWidth:640,dstHeight:480,bytes:expected.length,repeats:3,nativeExact:true,originalKernelUnchanged:true,fullBatchPipeline:false};
 }finally{runtime.destroyTexture(luma);runtime.destroyTexture(chroma);runtime.destroyBuffer(output);}
}
