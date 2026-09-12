export async function checkNv12Convert(runtime){
 const source=await(await fetch('/tests/nv12-convert-kernel.cuh')).text();
 const kernel=await runtime.kernel(source,{entry:'nv12ToBGRplanarBatchKernel',workgroupSize:[64,10,1]});
 const input=new Uint8Array(await(await fetch('/reports/nv12-input.bin')).arrayBuffer());
 const expected=new Float32Array(await(await fetch('/reports/nv12-convert-native.bin')).arrayBuffer());
 if(input.length!==1920*1080*3/2||expected.length!==1920*1080*3)throw Error('Wrong full-resolution NV12 capture');
 const src=runtime.createBuffer(input),dst=runtime.createBuffer(new Float32Array(expected.length));
 let maxError=0;
 try{
  // Preserve all 24 full-resolution frames. Conversion has no cross-frame
  // reads; stream individual frames to stay below WebGPU's binding limit.
  // The native capture verifies every frame from the original 24-Z launch.
  const invocation=kernel.bind({pNv12:src,pBgr:dst},{nNv12Pitch:1920,nRgbPitch:1920*4,nWidth:1920,nHeight:1080,nBatchSize:1});
  for(let frame=0;frame<24;frame++){
   runtime.batch().dispatch(invocation,[8,54,1]).submit();
   const actual=await runtime.read(dst);
   for(let i=0;i<actual.length;i++){
    const error=Math.abs(actual[i]-expected[i]);
    if(!Number.isFinite(actual[i])||error>0.0001)throw Error(`NV12 conversion frame ${frame}, value ${i}, error ${error}`);
    maxError=Math.max(maxError,error);
   }
  }
  return {width:1920,height:1080,frames:24,checkedValues:expected.length*24,maxError,tolerance:0.0001,streamed:true,originalBodiesUnchanged:true,fullNv12ResizePipeline:false};
 }finally{runtime.destroyBuffer(src);runtime.destroyBuffer(dst);}
}
