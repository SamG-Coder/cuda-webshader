// Original NVIDIA test0: texture padding, two R2Cs, modulation, then C2R.
export async function checkFFTConvolution(runtime){
 const source=await(await fetch('/tests/fft-convolution-kernels.cuh')).text(),results=[];
 const padKernel=await runtime.kernel(source,{entry:'padKernel_kernel',workgroupSize:[32,8,1]}),padData=await runtime.kernel(source,{entry:'padDataClampToBorder_kernel',workgroupSize:[32,8,1]}),modulate=await runtime.kernel(source,{entry:'modulateAndNormalize_kernel',workgroupSize:[256,1,1]});
 for(let test=0;test<2;test++){
  const width=test?2000:37,height=test?2000:19,fftW=test?2048:64,fftH=test?2048:32,n=fftW*fftH,sn=fftH*(fftW/2+1),load=async kind=>new Float32Array(await(await fetch(`/reports/fft-convolution-${test}-${kind}.bin`)).arrayBuffer()),input=await load('input'),weights=await load('kernel'),native=await load('output'),buffers=[],textures=[];
  const buffer=data=>{const b=runtime.createBuffer(data);buffers.push(b);return b;},texture=data=>{const t=runtime.createLinearTexture(data);textures.push(t);return t;};
  try{
   const di=buffer(input),dk=buffer(weights),pd=buffer(new Float32Array(n)),pk=buffer(new Float32Array(n)),ds=buffer(new Float32Array(sn*2)),ks=buffer(new Float32Array(sn*2)),ti=texture(input),tk=texture(weights),scalars={fftH,fftW,kernelH:7,kernelW:6,kernelY:3,kernelX:4},before=runtime.stats.readbackBytes;
   runtime.batch().dispatch(padKernel.bind({d_Dst:pk,d_Src:dk,texFloat:tk},scalars),[1,1,1]).dispatch(padData.bind({d_Dst:pd,d_Src:di,texFloat:ti},{...scalars,dataH:height,dataW:width}),[Math.ceil(fftW/32),Math.ceil(fftH/8),1]).submit();
   // Separate out-of-place R2Cs preserve the original compact real input rows.
   await runtime.realFFT2D(pk,ks,{width:fftW,height:fftH,realStride:fftW});
   await runtime.realFFT2D(pd,ds,{width:fftW,height:fftH,realStride:fftW});
   runtime.batch().dispatch(modulate.bind({d_Dst:ds,d_Src:ks},{dataSize:sn,c:1/n}),[Math.ceil(sn/256),1,1]).submit();
   await runtime.realFFT2D(ds,pd,{width:fftW,height:fftH,realStride:fftW,inverse:true});
   if(runtime.stats.readbackBytes!==before)throw Error('FFT convolution intermediate readback');
   const actual=await runtime.read(pd);let e2=0,r2=0,maxError=0,cpuE2=0,cpuR2=0,cpuMaxError=0;
   for(let i=0;i<n;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite convolution');const d=actual[i]-native[i];maxError=Math.max(maxError,Math.abs(d));e2+=d*d;r2+=native[i]*native[i];}
   // Independent spatial convolution, all pixels including every clamped edge.
   for(let y=0;y<height;y++)for(let x=0;x<width;x++){let expected=0;for(let ky=-3;ky<=3;ky++)for(let kx=-1;kx<=4;kx++){const dy=Math.max(0,Math.min(height-1,y+ky)),dx=Math.max(0,Math.min(width-1,x+kx));expected+=input[dy*width+dx]*weights[(3-ky)*6+4-kx];}const d=actual[y*fftW+x]-expected;cpuE2+=d*d;cpuR2+=2*expected*expected;cpuMaxError=Math.max(cpuMaxError,Math.abs(d));}
   const relativeL2=Math.sqrt(e2/r2),originalCpuRelativeL2=Math.sqrt(cpuE2/cpuR2);if(relativeL2>1e-6||originalCpuRelativeL2>=1e-6)throw Error('Convolution accuracy: '+JSON.stringify({relativeL2,originalCpuRelativeL2,maxError,cpuMaxError}));
   results.push({width,height,fftW,fftH,components:n,relativeL2,maxError,originalCpuRelativeL2,cpuMaxError,intermediateReadbackBytes:0});
  }finally{buffers.forEach(b=>runtime.destroyBuffer(b));textures.forEach(t=>runtime.destroyTexture(t));}
 }
 return {results,source:'Original NVIDIA test0 device bodies',input:'Original srand(2010), rand()%16 captured on Windows',passed:true};
}
