export async function checkDeferredPointers(runtime) {
  const source=`__global__ void fill(float*out) {
    int x=threadIdx.x;float*p=NULL;
    for(int i=blockIdx.z;i<24;i+=gridDim.z) {
      for(int channel=0;channel<3;channel++) {
        p=out+i*96+channel*32+x;
        *p=float(i*1000+channel*100+x);
      }
    }
  }`;
  const kernel=await runtime.kernel(source,{workgroupSize:[32,1,1]}),out=runtime.createBuffer(new Float32Array(24*96));
  try {
    runtime.batch().dispatch(kernel.bind({out},{}),[1,1,4]).submit();
    const actual=await runtime.read(out);
    for(let i=0;i<actual.length;i++) {
      const expected=Math.floor(i/96)*1000+Math.floor(i%96/32)*100+i%32;
      if(actual[i]!==expected)throw Error('Deferred pointer batch/channel mismatch at '+i);
    }
    const original=await(await fetch('/tests/nv12-bgr-kernel.cuh')).text();
    const resize=await runtime.kernel(original,{entry:'resizeBGRplanarBatchKernel',workgroupSize:[32,32,1]});
    const input=Float32Array.from({length:32*8*3*24},(_,i)=>i%32+Math.floor(i/32)*64);
    const texture=runtime.createTexture2D(input,{width:32,height:8*3*24,normalizedCoords:false,addressMode:'clamp-to-edge',filter:'linear'});
    const destination=runtime.createBuffer(new Float32Array(20*4*3*24).fill(-99));
    try {
      runtime.batch().dispatch(resize.bind({texSrc:texture,pDst:destination},{nDstPitch:20,nDstHeight:4,nSrcHeight:8,batch:24,scaleX:2,scaleY:2,cropX:0,cropY:0,cropW:32,cropH:8}),[1,1,4]).submit();
      const result=await runtime.read(destination),native=new Float32Array(await(await fetch('/reports/nv12-bgr-stage-native.bin')).arrayBuffer());
      if(native.length!==result.length)throw Error('Wrong native BGR stage size');
      for(let i=0;i<result.length;i++) {
        const x=i%20,row=Math.floor(i/20),sourceRow=Math.floor(row/4)*8+(row%4)*2;
        const expected=x>=16?-99:Math.max(0,x*2-.5)+Math.max(0,sourceRow-.5)*64;
        if(result[i]!==native[i]||result[i]!==expected)throw Error('Original BGR resize mismatch at '+i);
      }
      return {elements:actual.length,batches:24,channels:3,gridZ:4,exact:true,originalBgrStage:{values:result.length,nativeExact:true,paddingPreserved:true,sourceUnchanged:true,fullDefaultPipeline:false}};
    }finally{runtime.destroyTexture(texture);runtime.destroyBuffer(destination);}
  }finally{runtime.destroyBuffer(out);}
}
