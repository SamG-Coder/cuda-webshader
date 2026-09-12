export async function checkFFTLayouts(runtime,{dimensions,prefixName,roundtripLimit=2e-6}){
 const results=[],load=async(name)=>new Float32Array(await(await fetch('/reports/'+name+'.bin')).arrayBuffer());
 for(const [width,height] of dimensions){
  const stride=2*(Math.floor(width/2)+1),count=width*height,size=stride*height,prefix=prefixName+'-'+width+'x'+height,input=await load(prefix+'-input'),spectrum=await load(prefix+'-spectrum'),nativeInverse=await load(prefix+'-inverse'),guarded=new Float32Array(size+8).fill(-77);guarded.set(input);
  const buffer=runtime.createBuffer(guarded),nativeBuffer=runtime.createBuffer(spectrum),separate=runtime.createBuffer(new Float32Array(size+8).fill(-77));
  try{
   const before=runtime.stats.readbackBytes;await runtime.realFFT2D(buffer,buffer,{width,height,realStride:stride});if(runtime.stats.readbackBytes!==before)throw Error('Forward real FFT intermediate readback');
   const actual=await runtime.read(buffer);let maxForwardError=0,squaredError=0,squaredReference=0;
   for(let i=0;i<size;i++){if(!Number.isFinite(actual[i]))throw Error('Nonfinite spectrum');maxForwardError=Math.max(maxForwardError,Math.abs(actual[i]-spectrum[i]));squaredError+=(actual[i]-spectrum[i])**2;squaredReference+=spectrum[i]**2;}
   const relativeL2=Math.sqrt(squaredError/Math.max(squaredReference,1e-30));if(relativeL2>2e-6)throw Error('Native real spectrum relative error '+relativeL2);
   if(actual.slice(size).some(v=>v!==-77))throw Error('Forward real FFT guard changed');
   if(count<=32){for(let ky=0;ky<height;ky++)for(let kx=0;kx<Math.floor(width/2)+1;kx++){let re=0,im=0;for(let y=0;y<height;y++)for(let x=0;x<width;x++){const angle=-2*Math.PI*(kx*x/width+ky*y/height);re+=input[y*stride+x]*Math.cos(angle);im+=input[y*stride+x]*Math.sin(angle);}const i=ky*stride+kx*2;if(Math.abs(actual[i]-re)>1e-5||Math.abs(actual[i+1]-im)>1e-5)throw Error('Packed spectrum differs from independent DFT');}}
   await runtime.realFFT2D(nativeBuffer,separate,{width,height,realStride:stride,inverse:true});const inv=await runtime.read(separate);let inverseNormalizedError=0;
   for(let y=0;y<height;y++)for(let x=0;x<stride;x++){const v=inv[y*stride+x];if(x>=width){if(v!==-77)throw Error('Inverse FFT overwrote row padding');}else{if(!Number.isFinite(v))throw Error('Nonfinite inverse');inverseNormalizedError=Math.max(inverseNormalizedError,Math.abs(v-nativeInverse[y*width+x])/count);}}
   if(inv.slice(size).some(v=>v!==-77))throw Error('Inverse guard changed');if(inverseNormalizedError>2e-6)throw Error('Native inverse mismatch '+inverseNormalizedError);
   await runtime.realFFT2D(buffer,buffer,{width,height,realStride:stride,inverse:true});const roundtrip=await runtime.read(buffer);let normalizedRoundtripError=0;
   for(let y=0;y<height;y++)for(let x=0;x<width;x++)normalizedRoundtripError=Math.max(normalizedRoundtripError,Math.abs(roundtrip[y*stride+x]/count-input[y*stride+x]));
   if(normalizedRoundtripError>roundtripLimit||roundtrip.slice(size).some(v=>v!==-77))throw Error('Real FFT roundtrip failed '+width+'x'+height+': '+normalizedRoundtripError);
   results.push({width,height,realStride:stride,maxForwardError,relativeL2,inverseNormalizedError,normalizedRoundtripError,guardsIntact:true,intermediateReadbackBytes:0});
  }finally{[buffer,nativeBuffer,separate].forEach(b=>runtime.destroyBuffer(b));}
 }
 return results;
}
export async function checkRealFFT(runtime){
 const results=await checkFFTLayouts(runtime,{dimensions:[[1,1],[2,4],[8,4],[64,32],[512,512]],prefixName:'real-fft'}),load=async(name)=>new Float32Array(await(await fetch('/reports/'+name+'.bin')).arrayBuffer());
 const {executePipeline}=await import('/src/sandbox/pipeline.js'),{compile}=await import('/src/compiler/compiler.js'),source='__global__ void unused(float*out){out[0]=0.0f;}',artifacts=[],resources=[],textureResources=[];
 const plan={buffers:{input:{type:'f32',records:40,fill:'binary-f32',source:'/reports/real-fft-8x4-input.bin'},spectrum:{type:'vec2<f32>',records:20,fill:'zero'},output:{type:'f32',records:40,fill:'zero'}},steps:[{realFFT:{source:'input',target:'spectrum',width:8,height:4,realStride:10}},{realFFT:{source:'spectrum',target:'output',width:8,height:4,realStride:10,inverse:true}}],preview:{kind:'image',buffer:'output',format:'gray-f32',width:10,height:4}};
 try{
  const before=runtime.stats.readbackBytes,result=await executePipeline({plan,source,runtime,compiler:{},rootArtifact:compile(source),resources,textureResources,onArtifact:a=>artifacts.push(a.name)});
  if(runtime.stats.readbackBytes!==before)throw Error('Real FFT pipeline readback');
  const actual=await runtime.read(result.buffers.output),expected=await load('real-fft-8x4-inverse');for(let y=0;y<4;y++)for(let x=0;x<10;x++){if(x<8&&Math.abs(actual[y*10+x]-expected[y*8+x])>1e-5||x>=8&&actual[y*10+x]!==0)throw Error('Real FFT pipeline roundtrip mismatch');}
  if(!['realToComplex','packSpectrum','unpackSpectrum','complexToReal','forwardFftAxis','inverseFftAxis'].every(n=>artifacts.includes(n)))throw Error('Missing FFT comparison kernels');
 }finally{resources.forEach(b=>runtime.destroyBuffer(b));}
 return {results,pipelineVerified:true,artifacts,packedLayout:'height rows of floor(width/2)+1 complex values',inverseNormalization:'none'};
}

export async function checkLargeFFT(runtime){return {results:await checkFFTLayouts(runtime,{dimensions:[[2048,8],[8,2048],[2048,2048]],prefixName:'large-fft',roundtripLimit:4e-6}),maximumAxis:2048,maximumThreads:1024,roundtripLimit:4e-6};}
