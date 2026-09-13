import {checkChronoAdami} from './chrono-adami-gpu.js';
import {checkBoolStorage} from './bool-storage-gpu.js';
import {checkSwitch} from './switch-gpu.js';
import {checkChronoReorder} from './chrono-reorder-gpu.js';
import {checkChronoSelected} from './chrono-selected-gpu.js';
import {checkChronoCompact} from './chrono-compact-gpu.js';
import {checkChronoActivity} from './chrono-activity-gpu.js';
import {checkChronoCounterTime} from './chrono-counter-time-gpu.js';
import {checkCapturedPoints} from './captured-points-gpu.js';
import {checkChronoNeighbors} from './chrono-neighbors-gpu.js';
import {checkChronoTypes} from './chrono-types-gpu.js';
import {checkChronoHash} from './chrono-hash-gpu.js';
import {checkChronoSearch} from './chrono-search-gpu.js';
import {checkQuadtreeRoot} from './quadtree-root-gpu.js';
import {checkBezierScheduled} from './bezier-scheduled-gpu.js';
import {checkBezierParent} from './bezier-parent-gpu.js';
import {checkBezierChild} from './bezier-cdp-gpu.js';
import {checkDeviceHeap} from './device-heap-gpu.js';
import {checkLayered} from './layered-gpu.js';
import {checkDeferredPointers} from './deferred-pointers-gpu.js';
import {checkNv12Convert} from './nv12-convert-gpu.js';
import {checkNv12Resize} from './nv12-resize-gpu.js';
import {checkPathtracerStorage} from './pathtracer-storage-gpu.js';
import {checkPathtracerMaterials} from './pathtracer-material-gpu.js';
import {checkPathtracerCamera} from './pathtracer-camera-gpu.js';
import {checkPersistentList} from './pathtracer-list-gpu.js';
import {checkPersistentObjects} from './pathtracer-persistent-gpu.js';
import {checkPathtracerValues} from './pathtracer-value-gpu.js';
import {checkCubemap} from './cubemap-gpu.js';
import {checkWalshFull} from './walsh-full-gpu.js';
import {checkBinomial} from './binomial-gpu.js';
import {checkDeviceGlobals} from './device-globals-gpu.js';
import {checkOddEven} from './odd-even-gpu.js';
import {checkSortingNetworks} from './sorting-networks-gpu.js';
import {checkSobol} from './sobol-gpu.js';
import {checkQuasirandom} from './quasirandom-gpu.js';
import {checkDxtCompress} from './dxt-compress-gpu.js';
import {checkDxtColors} from './dxt-colors-gpu.js';
import {checkDxtEvaluate} from './dxt-evaluate-gpu.js';
import {checkRoundEven} from './round-even-gpu.js';
import {checkFFTCustomFull} from './fft-custom-full-gpu.js';
import {checkSincos} from './sincos-gpu.js';
import {checkFFTCustom} from './fft-custom-gpu.js';
import {checkFFTConvolution} from './fft-convolution-gpu.js';
import {checkLinearFloat,checkLinearFloatCopy} from './linear-float-gpu.js';
import {checkOpticalFlow} from './optical-flow-gpu.js';
import {checkOpticalJacobi} from './optical-jacobi-gpu.js';
import {checkStereo} from './stereo-gpu.js';
import {checkPtxSad} from './ptx-sad-gpu.js';
import {checkFluidsSolver} from './fluids-solver-gpu.js';
import {checkRealFFT,checkLargeFFT} from './real-fft-gpu.js';
import {checkFluidsPitched} from './fluids-pitched-gpu.js';
import {checkSizeValues} from './size-values-gpu.js';
import {checkFluids} from './fluids-gpu.js';
import {checkSmokePipeline} from './smoke-pipeline-gpu.js';
import {checkFloatSortPairs} from './float-sort-gpu.js';
import {checkSmokeIntegration} from './smoke-integration-gpu.js';
import {checkSmokeNoise} from './smoke-noise-gpu.js';
import {checkVolumePreintegrated} from './volume-preintegrated-gpu.js';
import {checkFloat64} from './float64-gpu.js';
import {checkVolumeTransfer} from './volume-transfer-gpu.js';
import {checkFloat4Surface} from './float4-surface-gpu.js';
import {checkParticleSimulation} from './particle-simulation-gpu.js';
import {checkSortPairs} from './sort-pairs-gpu.js';
import {checkParticleCollision} from './particle-collision-gpu.js';
import {checkFFT} from './fft-gpu.js';
import {checkOcean} from './ocean-gpu.js';
import {checkDctOptimized} from './dct-optimized-gpu.js';
import {checkDctFloat} from './dct-float-gpu.js';
import {checkDenoising} from './denoising-gpu.js';
import {checkSobelShared} from './sobel-shared-gpu.js';
import {checkSobelImages} from './sobel-image-gpu.js';
import {checkShortValues} from './short-values-gpu.js';
import {checkSobelNeighborhoods} from './sobel-neighborhoods-gpu.js';
import {checkByteStores} from './byte-stores-gpu.js';
import {checkBoxFilter} from './box-filter-gpu.js';
import {checkMarchingVolume} from './marching-volume-gpu.js';
import {checkSharedPointers} from './shared-pointer-gpu.js';
import {checkMarchingPipeline} from './marching-pipeline-gpu.js';
import {checkExclusiveScan} from './exclusive-scan-gpu.js';
import {checkMarchingTriangles} from './marching-triangles-gpu.js';
import {checkSharedVectorReferences} from './shared-vector-reference-gpu.js';
import {checkMarchingClassify} from './marching-classify-gpu.js';
import {checkVectorReferences} from './vector-reference-gpu.js';
import {checkVectorByteArguments} from './vector-byte-arguments-gpu.js';
import {checkLinearTextures} from './linear-texture-gpu.js';
import {checkVolumeFilter} from './volume-filter-gpu.js';
import {checkMandelbrot} from './mandelbrot-gpu.js';
import {kernelSource} from '../src/sandbox/import.js';
import {checkBilateral} from './bilateral-filter-gpu.js';
import {checkPostProcess} from './postprocess-gl-gpu.js';
import {checkBicubic} from './bicubic-texture-gpu.js';
import {checkConvolutionTexture} from './convolution-texture-gpu.js';
import {checkSurfaceWrite} from './surface-write-gpu.js';
import {checkTexture2D} from './texture2d-gpu.js';
import {volumeTransfer,volumeCase} from './volume-render-cases.js';
import {transferCoordinates,transferValues,transferSample} from './transfer-texture-reference.js';
import {texture3dInitial,sampleTexture3D} from './texture3d-reference.js';
import {gaussianCoefficients,gaussianInitial,gaussianColumns,gaussianTranspose,gaussianPixelError} from './recursive-gaussian-reference.js';
import {haarInitial,haarReference} from './haar-reference.js';
import {prepareReduction} from '../src/runtime/operations.js';import {kernelOptions} from '../src/kernels.js';
import {makeCases,compareArrays} from './cases.js';
import {nbodyInitial,nbodyStep} from './nbody-reference.js';
import {fdtdInitial,fdtdStep,fdtdCoefficients} from './fdtd3d-reference.js';
import {separableInitial,separableReference,separableCoefficients} from './convolution-separable-reference.js';
export async function runGpuSuite(runtime,sources,{onCase=()=>{}}={}){
  const results=[],start=performance.now();
  const run=async(name,action)=>{const t=performance.now();let result;try{result={name,pass:true,...await action()};}catch(error){result={name,pass:false,error:String(error.stack||error)};}result.wallMs=performance.now()-t;results.push(result);onCase(result,results.length);};
  for(const c of makeCases())await run(c.name,async()=>{
    const buffers={};try{
      for(const [name,data]of Object.entries(c.buffers))buffers[name]=runtime.createBuffer(data,{label:`${c.name}: ${name}`});
      const kernel=await runtime.kernel(sources[c.id],kernelOptions(c.id));
      const invocation=kernel.bind(buffers,c.scalars);runtime.batch().dispatch(invocation,c.groups).submit();
      const checks={};for(const [name,expected]of Object.entries(c.expected)){
        const actual=await runtime.read(buffers[name],expected.constructor);checks[name]=compareArrays(actual,expected,c.tolerance);
        if(!checks[name].pass)throw new Error(`${name} differs from independent reference: ${JSON.stringify(checks[name])}`);
      }
      return {checks};
    }finally{for(const resource of Object.values(buffers))runtime.destroyBuffer(resource);}
  });
  await run('Two parameter versions in one batch retain their own snapshots',async()=>{
    const values=runtime.createBuffer(new Float32Array(2));
    try{
      const kernel=await runtime.kernel('__global__ void stamp(float* values,unsigned int slot,float value){if(threadIdx.x==0u)values[slot]=value;}',{workgroupSize:[4,1,1]});
      const inv=kernel.bind({values},{slot:0,value:11});const batch=runtime.batch();batch.dispatch(inv,[1]);inv.setScalars({slot:1,value:22});batch.dispatch(inv,[1]);batch.submit();
      const output=await runtime.read(values);if(output[0]!==11||output[1]!==22)throw new Error(`Uniform snapshot regression: ${output}`);
    }finally{runtime.destroyBuffer(values);}
  });
  await run('Reusable hierarchical reduction, including empty and singleton inputs',async()=>{
    for(const n of [0,1,4099]){
      const data=Float32Array.from({length:n},(_,i)=>(i%13-6)*0.125),expected=data.reduce((a,b)=>a+b,0);
      const input=runtime.createBuffer(data);let plan;
      try{
        plan=await prepareReduction(runtime,sources,{input,n});
        plan.encode(runtime.batch()).submit();
        const result=await runtime.read(plan.output);
        if(result[0]!==expected)throw new Error(`Reduction n=${n}: ${result[0]} != ${expected}`);
      }finally{plan?.dispose();runtime.destroyBuffer(input);}
    }
  });
  await run('Device helper templates: deduction, specialization, nested types, references and vectors',async()=>{
    const source=await (await fetch('/tests/nbody-rsqrt.cuh')).text()+'\n'+await (await fetch('/tests/helper-templates.cu')).text();
    const kernel=await runtime.kernel(source,{workgroupSize:[128,1,1]});
    for(const n of [1,129,1025]){
      const out=runtime.createBuffer(new Float32Array(n*4+16).fill(-12345)),bits=runtime.createBuffer(new Uint32Array(n+16).fill(0xdeadbeef));
      try{
        runtime.batch().dispatch(kernel.bind({out,bits},{n}),[Math.ceil(n/128)]).submit();
        const values=await runtime.read(out),integers=await runtime.read(bits,Uint32Array);
        for(let i=0;i<n;i++){const x=(i%13-6)*0.25,expected=[x*x+x,x*3,2,x+1.5],v=(0x80000000+i)>>>0;
          for(let j=0;j<4;j++)if(values[i*4+j]!==expected[j])throw Error('Template float mismatch at '+i);
          if(integers[i]!==((Math.imul(v,v)+v)>>>0))throw Error('Template unsigned mismatch at '+i);
        }
        if(!values.slice(n*4).every(v=>v===-12345)||!integers.slice(n).every(v=>v===0xdeadbeef))throw Error('Template guard changed');
      }finally{await runtime.idle();runtime.destroyBuffer(out);runtime.destroyBuffer(bits);}
    }
  });
  await run('NVIDIA N-body vector traits resolve in kernel buffers and device helpers',async()=>{
    const source=await (await fetch('/tests/nbody-vector-traits.cuh')).text()+'\n'+await (await fetch('/tests/type-traits.cu')).text();
    const kernel=await runtime.kernel(source,{entry:'traitPositions<float>',workgroupSize:[128,1,1]});
    for(const n of [1,129,1025]){
      const data=Float32Array.from({length:(n+4)*4},(_,i)=>i<n*4?(i%17-8)*0.125:-12345),positions=runtime.createBuffer(data);
      try{
        runtime.batch().dispatch(kernel.bind({positions},{n,dt:0.25}),[Math.ceil(n/128)]).submit();
        const output=await runtime.read(positions),delta=[0.125,-0.0625,0.25,0];
        for(let i=0;i<data.length;i++)if(output[i]!==data[i]+(i<n*4?delta[i%4]:0))throw Error('N-body trait layout/update mismatch at '+i);
      }finally{await runtime.idle();runtime.destroyBuffer(positions);}
    }
  });
  await run('NVIDIA N-body interaction reads constant uniforms with per-dispatch snapshots',async()=>{
    const files=['nbody-vector-traits.cuh','nbody-rsqrt.cuh','nbody-interaction.cuh','constant-globals.cu'],source=(await Promise.all(files.map(async f=>await(await fetch('/tests/'+f)).text()))).join('\n');
    const kernel=await runtime.kernel(source,{workgroupSize:[128,1,1]});
    for(const n of [1,129,1025]){
      const out=runtime.createBuffer(new Float32Array((n*3+4)*4).fill(-12345)),softening=[0,0.25,2];
      try{
        const invocation=kernel.bind({out},{n,offset:0}),batch=runtime.batch();
        for(let pass=0;pass<3;pass++){invocation.setScalars({n,offset:pass*n,...(pass?{'constant.softeningSquared':softening[pass]}:{})});batch.dispatch(invocation,[Math.ceil(n/128)]);}batch.submit();
        const output=await runtime.read(out);
        for(let pass=0;pass<3;pass++)for(let i=0;i<n;i++){
          const r=[2-(i%8)*0.125,-1-(i%3)*0.25,0.5-(i%5)*0.125],s=1.5/Math.pow(r.reduce((sum,x)=>sum+x*x,softening[pass]),1.5),expected=[...r.map(x=>x*s),1];
          for(let c=0;c<4;c++){const actual=output[(pass*n+i)*4+c];if(!Number.isFinite(actual)||Math.abs(actual-expected[c])>3e-6+2e-6*Math.abs(expected[c]))throw Error('N-body interaction mismatch');}
        }
        if(!output.slice(n*12).every(v=>v===-12345))throw Error('N-body interaction guard changed');
      }finally{await runtime.idle();runtime.destroyBuffer(out);}
    }
  });
  await run('Static and dynamic shared helpers coordinate lanes through nested calls',async()=>{
    for(const dynamic of [false,true]){
      const source=await(await fetch('/tests/'+(dynamic?'shared-helpers-dynamic.cu':'shared-helpers.cu'))).text();
      for(const threads of [32,128]){const groups=3,n=threads*groups,out=runtime.createBuffer(new Float32Array(n+16).fill(-12345));
        try{const kernel=await runtime.kernel(source,{workgroupSize:[threads,1,1],...(dynamic?{sharedMemoryBytes:threads*4}:{})});runtime.batch().dispatch(kernel.bind({out},{}),[groups]).submit();const actual=await runtime.read(out);
          for(let i=0;i<n;i++){const reversed=Math.floor(i/threads)*threads+threads-1-i%threads,expected=dynamic?reversed:reversed*1.125+threads+groups;if(actual[i]!==expected)throw Error('Shared helper mismatch at '+i);}
          if(!actual.slice(n).every(v=>v===-12345))throw Error('Shared helper guard changed');
        }finally{await runtime.idle();runtime.destroyBuffer(out);}
      }
    }
  });
  await run('Storage pointer helpers preserve nested offsets, vectors, atomics, aliases and shared storage',async()=>{
    const source=await(await fetch('/tests/helper-pointers.cu')).text();
    for(const threads of [32,128]){
      const n=threads*3,input=Float32Array.from({length:n+1},(_,i)=>i*0.125),vectors=Float32Array.from({length:n*4},(_,j)=>[Math.floor(j/4)*0.25,-Math.floor(j/4)*0.5,2,1][j%4]);
      const buffers={input:runtime.createBuffer(input),other:runtime.createBuffer(new Float32Array(n).fill(999)),vectors:runtime.createBuffer(vectors),out:runtime.createBuffer(new Float32Array(n+17).fill(-12345)),copied:runtime.createBuffer(new Float32Array((n+4)*4).fill(-12345)),counts:runtime.createBuffer(new Int32Array(1))};
      try{
        const kernel=await runtime.kernel(source,{workgroupSize:[threads,1,1]});runtime.batch().dispatch(kernel.bind(buffers,{}),[3]).submit();
        const out=await runtime.read(buffers.out),copied=await runtime.read(buffers.copied),counts=await runtime.read(buffers.counts,Int32Array);
        if(counts[0]!==n||out[n]!==4)throw Error('Pointer atomic or alias mismatch');
        for(let i=0;i<n;i++)if(out[i]!==input[i]+input[i+1])throw Error('Pointer offset/shared reuse mismatch at '+i);
        for(let i=0;i<vectors.length;i++)if(copied[i]!==vectors[i])throw Error('Pointer vector mismatch at '+i);
        if(!out.slice(n+1).every(v=>v===-12345)||!copied.slice(n*4).every(v=>v===-12345))throw Error('Pointer guard changed');
      }finally{await runtime.idle();for(const buffer of Object.values(buffers))runtime.destroyBuffer(buffer);}
    }
  });
  await run('Vector aggregate initializers preserve partial zero filling and dependent helper types',async()=>{
    const source=await(await fetch('/tests/vector-initializers.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[128,1,1]});
    for(const n of [1,129,1025]){const out=runtime.createBuffer(new Float32Array(n*12+16).fill(-12345));try{
      runtime.batch().dispatch(kernel.bind({out},{n}),[Math.ceil(n/128)]).submit();const actual=await runtime.read(out);
      for(let i=0;i<n;i++){const expected=[0,0,0,0,i+5,0,0,0,1,2,3,4];for(let j=0;j<12;j++)if(actual[i*12+j]!==expected[j])throw Error('Vector initializer mismatch at '+i);}
      if(!actual.slice(n*12).every(v=>v===-12345))throw Error('Vector initializer guard changed');
    }finally{await runtime.idle();runtime.destroyBuffer(out);}}
  });
  await run('Original NVIDIA shared-memory conversion wrapper exchanges float4 tiles',async()=>{
    const source=await(await fetch('/tests/nbody-shared-memory.cuh')).text()+'\n'+await(await fetch('/tests/shared-wrapper.cu')).text();
    for(const threads of [32,128]){const n=threads*3,data=Float32Array.from({length:n*4},(_,i)=>i*0.125),input=runtime.createBuffer(data),output=runtime.createBuffer(new Float32Array(n*4+16).fill(-12345));try{
      const kernel=await runtime.kernel(source,{workgroupSize:[threads],sharedMemoryBytes:threads*16});runtime.batch().dispatch(kernel.bind({input,output},{}),[3]).submit();const actual=await runtime.read(output);
      for(let i=0;i<n;i++)for(let j=0;j<4;j++)if(actual[i*4+j]!==data[(Math.floor(i/threads)*threads+threads-i%threads-1)*4+j])throw Error('Shared wrapper tile mismatch');
      if(!actual.slice(n*4).every(v=>v===-12345))throw Error('Shared wrapper guard changed');
    }finally{await runtime.idle();runtime.destroyBuffer(input);runtime.destroyBuffer(output);}}
  });
  await run('Original NVIDIA N-body advances three steps with an enforced whole-block contract',async()=>{
    const files=['nbody-vector-traits.cuh','nbody-rsqrt.cuh','nbody-interaction.cuh','nbody-shared-memory.cuh','nbody-integrate.cuh'];
    const source='namespace cg = cooperative_groups;\n'+(await Promise.all(files.map(async f=>await(await fetch('/tests/'+f)).text()))).join('\n');
    for(const threads of [32,128])for(const blocks of [1,3,4]){
      const n=threads*blocks,dt=Math.fround(0.002),damping=Math.fround(0.999),softening=0.125;let expected=nbodyInitial(n);
      let oldPos=runtime.createBuffer(expected.positions),newPos=runtime.createBuffer(new Float32Array((n+4)*4).fill(-12345));const vel=runtime.createBuffer(expected.velocities);
      try{const kernel=await runtime.kernel(source,{entry:'integrateBodies<float>',workgroupSize:[threads],sharedMemoryBytes:threads*16,fullWorkgroups:['deviceNumBodies']});
        for(let step=0;step<3;step++){
          expected=nbodyStep(expected.positions,expected.velocities,n,dt,damping,softening);
          runtime.batch().dispatch(kernel.bind({newPos,oldPos,vel},{deviceOffset:0,deviceNumBodies:n,deltaTime:dt,damping,numTiles:blocks,'constant.softeningSquared':softening}),[blocks]).copy(newPos,oldPos).submit();
          const positions=await runtime.read(newPos),velocities=await runtime.read(vel);
          for(let i=0;i<positions.length;i++)if(!Number.isFinite(positions[i])||!Number.isFinite(velocities[i])||Math.abs(positions[i]-expected.positions[i])>3e-5||Math.abs(velocities[i]-expected.velocities[i])>3e-5)throw Error(`N-body reference mismatch n=${n} step=${step} index=${i}`);

        }
      }finally{await runtime.idle();runtime.destroyBuffer(oldPos);runtime.destroyBuffer(newPos);runtime.destroyBuffer(vel);}
    }
  });
  await run('Original NVIDIA N-body early return remains rejected before a helper barrier',async()=>{
    const files=['nbody-vector-traits.cuh','nbody-rsqrt.cuh','nbody-interaction.cuh','nbody-shared-memory.cuh','nbody-integrate.cuh'];
    const source='namespace cg = cooperative_groups;\n'+(await Promise.all(files.map(async f=>await(await fetch('/tests/'+f)).text()))).join('\n');
    try{await runtime.kernel(source,{entry:'integrateBodies<float>',workgroupSize:[128],sharedMemoryBytes:2048});}catch(error){if(/uniform control flow/.test(error.message))return {expectedRejection:true,reason:'Early return depends on the lane index before a shared-memory barrier.'};throw error;}
    throw Error('Expected N-body barrier uniformity rejection; review the execution contract before enabling this sample.');
  });
  await run('Constant integer arrays retain defaults, high bits and per-dispatch snapshots',async()=>{
    const kernel=await runtime.kernel('__constant__ unsigned int table[2]={4294967295u,2147483648u};__device__ unsigned int value(unsigned int i){return table[i];}__global__ void k(unsigned int* out,unsigned int offset){out[offset+threadIdx.x]=value(threadIdx.x);}',{workgroupSize:[2]});const out=runtime.createBuffer(new Uint32Array(4));
    try{const inv=kernel.bind({out},{offset:0}),batch=runtime.batch().dispatch(inv,[1]);inv.setScalars({offset:2,'constant.table[0]':7,'constant.table[1]':9});batch.dispatch(inv,[1]).submit();const actual=await runtime.read(out,Uint32Array);if(String(actual)!=='4294967295,2147483648,7,9')throw Error('Constant array values or snapshots changed');}finally{await runtime.idle();runtime.destroyBuffer(out);}
  });
  await run('RGBA 2D linear sampling and transfer helper use distinct coordinate modes',async()=>{const source='__device__ float4 sample(cudaTextureObject_t t,float x,float y){return tex2D<float4>(t,x,y);}__global__ void k(float4* out,cudaTextureObject_t image,cudaTextureObject_t table){out[0]=sample(image,1.f,1.f);out[1]=tex1D<float4>(table,.5f);}',kernel=await runtime.kernel(source,{workgroupSize:[1]}),image=runtime.createTexture2D(Float32Array.from({length:16},(_,i)=>i/16),{width:2,height:2,format:'rgba32float',filter:'linear',normalizedCoords:false,addressMode:'clamp-to-edge'}),table=runtime.createTexture1D(Float32Array.from([1,2,3,4])),out=runtime.createBuffer(32);try{runtime.batch().dispatch(kernel.bind({out,image,table}),[1]).submit();const actual=await runtime.read(out),expected=[.375,.4375,.5,.5625,1,2,3,4];if(actual.some((v,i)=>v!==expected[i]))throw Error('RGBA linear or transfer coordinates mismatch');}finally{runtime.destroyTexture(image);runtime.destroyTexture(table);runtime.destroyBuffer(out);}});
  await run('Original NVIDIA Mandelbrot and Julia frames match native CUDA',()=>checkMandelbrot(runtime));
  await run('Packed destinations match native CUDA C++17 evaluation ordering',async()=>{const source=await(await fetch('/tests/packed-storage.cu')).text(),kernel=await runtime.kernel(source,{entry:'orderedStores',workgroupSize:[1]}),out=runtime.createBuffer(new Uint32Array(3).fill(0x04030201));try{runtime.batch().dispatch(kernel.bind({out},{}),[1]).submit();const actual=await runtime.read(out,Uint32Array);if(actual[0]!==0x04030201||actual[1]!==0x04030401||actual[2]!==0x04030201)throw Error('Packed evaluation ordering mismatch');}finally{runtime.destroyBuffer(out);}});
  await run('3D surface writes match native CUDA and reject oversized XYZ dispatches',async()=>{const source=await(await fetch('/tests/surface3d.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[4,4,2]}),dst=runtime.createTexture3D(null,{width:8,height:8,depth:4,storage:true,format:'r32float',filter:'nearest'}),staging=runtime.device.createBuffer({size:256*8*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const invocation=kernel.bind({dst},{});for(const groups of [[3,2,2],[2,3,2],[2,2,3]]){const b=runtime.batch();try{let rejected=false;try{b.dispatch(invocation,groups);}catch(e){if(!/Surface dispatch/.test(e.message))throw e;rejected=true;}if(!rejected)throw Error('Oversized 3D surface dispatch accepted');}finally{b.discard();}}runtime.batch().dispatch(invocation,[2,2,2]).submit();const encoder=runtime.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:dst.gpuTexture},{buffer:staging,bytesPerRow:256,rowsPerImage:8},[8,8,4]);runtime.device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);const values=new Float32Array(staging.getMappedRange());for(let z=0;z<4;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){const i=(z*8+y)*8+x;if(values[(z*8+y)*64+x]!==i*.125-16)throw Error('3D surface voxel mismatch '+i);}staging.unmap();return {voxels:256,nativeExact:true,oversizedXYZRejected:true};}finally{staging.destroy();runtime.destroyTexture(dst);}});
  await run('Imported device type aliases match native CUDA',async()=>{const source=kernelSource('#include <cuda_runtime.h>\nstruct Host {cudaArray* array;};\n'+await(await fetch('/tests/type-aliases.cu')).text()).source,kernel=await runtime.kernel(source,{workgroupSize:[1]}),out=runtime.createBuffer(new Uint32Array(4));try{runtime.batch().dispatch(kernel.bind({out},{}),[1]).submit();const actual=await runtime.read(out,Uint32Array);if([4,6,5,4].some((v,i)=>actual[i]!==v))throw Error('Alias mismatch');return {nativeExact:true,values:Array.from(actual)};}finally{runtime.destroyBuffer(out);}});
  await run('Exact double-literal byte scaling matches native rounding boundaries',async()=>{const source=await(await fetch('/tests/volume-convert-kernel.cuh')).text(),values=Float32Array.from({length:255},(_,i)=>(i+1)/65535),native=new Uint32Array(await(await fetch('/reports/exact-byte-scale-native.bin')).arrayBuffer()),kernel=await runtime.kernel(source,{entry:'scaleProbe',defines:{__CUDACC__:1},workgroupSize:[64]}),input=runtime.createBuffer(values),output=runtime.createBuffer(new Uint32Array(255));try{runtime.batch().dispatch(kernel.bind({input,output},{n:255}),[4]).submit();const actual=await runtime.read(output,Uint32Array);if(native.length!==255)throw Error('Native scale size');for(let i=0;i<255;i++)if(actual[i]!==native[i]||actual[i]!==Math.trunc(values[i]*65535)||actual[i]===Math.trunc(Math.fround(values[i]*65535)))throw Error('Exact scale mismatch '+i);return {values:255,nativeExact:true,floatProductShortcutDifferences:255};}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('Original NVIDIA static byte conversion matches native at rounding boundaries',async()=>{const source=await(await fetch('/tests/volume-convert-kernel.cuh')).text(),inputValues=new Float32Array(await(await fetch('/reports/volume-convert-input.bin')).arrayBuffer()),native=new Uint32Array(await(await fetch('/reports/volume-convert-native.bin')).arrayBuffer()),kernel=await runtime.kernel(source,{entry:'conversionProbe',defines:{__CUDACC__:1},workgroupSize:[64]}),input=runtime.createBuffer(inputValues),output=runtime.createBuffer(new Uint32Array(inputValues.length+16).fill(0xdeadbeef));try{runtime.batch().dispatch(kernel.bind({input,output},{n:inputValues.length}),[Math.ceil(inputValues.length/64)]).submit();const actual=await runtime.read(output,Uint32Array);if(native.length!==actual.length)throw Error('Native byte conversion size');let roundedDifferences=0;for(let i=0;i<actual.length;i++){const v=Math.max(0,Math.min(1,inputValues[i])),expected=i<inputValues.length?Math.trunc(v*255):0xdeadbeef;if(actual[i]!==expected||actual[i]!==native[i])throw Error('Byte conversion mismatch '+i);if(i<inputValues.length&&Math.trunc(Math.fround(v*255))!==expected)roundedDifferences++;}return {values:inputValues.length,nativeExact:true,roundedFloatProductDifferences:roundedDifferences};}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('cudaExtent size comparisons match native CUDA including signed negatives',async()=>{const source=await(await fetch('/tests/cuda-extent.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[32]}),values=new Int32Array([-2147483648,-1,0,1,7,8,2147483647]),input=runtime.createBuffer(values),output=runtime.createBuffer(new Uint32Array(56)),native=new Uint32Array(await(await fetch('/reports/cuda-extent-native.bin')).arrayBuffer());try{if(native.length!==168)throw Error('Native extent size');for(const [scenario,width]of [0,8,0xffffffff].entries()){runtime.batch().dispatch(kernel.bind({input,output},{'size.width':width,'size.height':3,'size.depth':4,n:7}),[1]).submit();const actual=await runtime.read(output,Uint32Array);for(let i=0;i<56;i++)if(actual[i]!==native[scenario*56+i])throw Error('Extent mismatch '+scenario+':'+i);}return {comparisons:168,nativeExact:true,unsigned64Promotion:true};}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('Float4 constant array helper reads match native CUDA',async()=>{const source=await(await fetch('/tests/vector-constants.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[64]}),values={};for(let i=0;i<125;i++)for(const [c,v]of Object.entries({x:i*.125,y:-i*.25,z:3.5,w:.25}))values['constant.weights['+i+'].'+c]=v;const output=runtime.createBuffer(new Float32Array(125));try{runtime.batch().dispatch(kernel.bind({output},values),[2]).submit();const actual=await runtime.read(output);for(let i=0;i<125;i++)if(actual[i]!==3.75-i*.125)throw Error('Vector constant mismatch '+i);return {records:125,components:500,nativeExact:true};}finally{runtime.destroyBuffer(output);}});
  await run('Complete implicit marching-cubes GPU pipeline matches native CUDA',()=>checkMarchingPipeline(runtime));
  await run('Hierarchical exclusive scan matches native CUDA',()=>checkExclusiveScan(runtime));
  await run('Original shared marching-cubes triangle generation matches native CUDA',()=>checkMarchingTriangles(runtime));
  await run('Full fluids solver compared with native CUDA and cuFFT',()=>checkFluidsSolver(runtime));
  await run('Real FFT packed transforms match native cuFFT',()=>checkRealFFT(runtime));
  await run('Original fluids pitched kernels match native CUDA',()=>checkFluidsPitched(runtime));
  await run('2048-point real FFT axes match native cuFFT',()=>checkLargeFFT(runtime));
  await run('Sincos and first-set-bit intrinsics match native CUDA',()=>checkSincos(runtime));
  await run('Original NVIDIA scalar layered textures match native CUDA',()=>checkLayered(runtime));
  await run('Deferred pointer assignments preserve batch and channel offsets',()=>checkDeferredPointers(runtime));
  await run('Original full-resolution NV12 conversion matches native CUDA',()=>checkNv12Convert(runtime));
  await run('Original single-frame NV12 resize matches native byte output',()=>checkNv12Resize(runtime));
  await run('Original CUDA RNG buffers and downcast cleanup match native',()=>checkPathtracerStorage(runtime));
  await run('Original CUDA materials scatter through virtual dispatch',()=>checkPathtracerMaterials(runtime));
  await run('Original CUDA camera and cuRAND XORWOW match native',()=>checkPathtracerCamera(runtime));
  await run('Original CUDA hitable_list captures live object buffers',()=>checkPersistentList(runtime));
  await run('Persistent CUDA sphere objects survive separate create/trace/free submissions',()=>checkPersistentObjects(runtime));
  await run('Path tracer value-class constructors and accessors match native CUDA',()=>checkPathtracerValues(runtime));
  await run('NVIDIA cubemap faces and edge filtering match native CUDA',()=>checkCubemap(runtime));
  await run('Complete NVIDIA Walsh convolution matches native CUDA',()=>checkWalshFull(runtime));
  await run('Original NVIDIA binomial options match all native outputs',()=>checkBinomial(runtime));
  await run('Device globals persist across dispatches and support helper atomics',()=>checkDeviceGlobals(runtime));
  await run('Complete NVIDIA odd-even sorting matches native CUDA',()=>checkOddEven(runtime));
  await run('Complete NVIDIA bitonic sorting matches native CUDA',()=>checkSortingNetworks(runtime));
  await run('Full Sobol sequence matches native CUDA',()=>checkSobol(runtime));
  await run('Full quasirandom sequence matches native CUDA',()=>checkQuasirandom(runtime));
  await run('DXT full image matches native CUDA',()=>checkDxtCompress(runtime));
  await run('DXT tiled colour stage matches native CUDA',()=>checkDxtColors(runtime));
  await run('DXT endpoint fitting matches native CUDA',()=>checkDxtEvaluate(runtime));
  await run('rintf matches native bits including halfway ties',()=>checkRoundEven(runtime));
  await run('Complete custom FFT convolution variants match native CUDA',()=>checkFFTCustomFull(runtime));
  await run('Original custom FFT convolution stages match native CUDA',()=>checkFFTCustom(runtime));
  await run('Original full-size FFT convolution matches native CUDA',()=>checkFFTConvolution(runtime));
  await run('GPU copies to multi-row float linear textures match native',()=>checkLinearFloatCopy(runtime));
  await run('Raw float linear texture records match native',()=>checkLinearFloat(runtime));
  await run('Complete optical flow matches original native pipeline',()=>checkOpticalFlow(runtime));
  await run('Original optical flow Jacobi matches 500 native iterations',()=>checkOpticalJacobi(runtime));
  await run('Original stereo disparity matches native pixels',()=>checkStereo(runtime));
  await run('Original NVIDIA inline PTX SAD matches native',()=>checkPtxSad(runtime));
  await run('size_t launch values preserve native 64-bit arithmetic',()=>checkSizeValues(runtime));
  await run('Original fluids kernels: native projection and bounded texture filtering',()=>checkFluids(runtime));
  await run('Smoke sandbox pipeline matches native integration and depth sorting',()=>checkSmokePipeline(runtime));
  await run('Float key sorting preserves IEEE bits and matches native smoke depths',()=>checkFloatSortPairs(runtime));
  await run('Original smoke integration and depth match native Thrust',()=>checkSmokeIntegration(runtime));
  await run('Original smoke float4 noise sampling matches native CUDA',()=>checkSmokeNoise(runtime));
  await run('Original preintegrated volume tables and ray marching match native CUDA',()=>checkVolumePreintegrated(runtime));
  await run('Integer-limb binary64 arithmetic matches native CUDA bits',()=>checkFloat64(runtime));
  await run('Original volume transfer integration matches native CUDA',()=>checkVolumeTransfer(runtime));
  await run('Float4 1D surfaces match native CUDA with checked extents',()=>checkFloat4Surface(runtime));
  await run('Original particle functor and collision simulation match 64 native steps',()=>checkParticleSimulation(runtime));
  await run('GPU stable key/value sort preserves unsigned keys and guards',()=>checkSortPairs(runtime));
  await run('Original NVIDIA particle collision stages match native CUDA',()=>checkParticleCollision(runtime));
  await run('GPU inverse FFT matches independent DFT and preserves guards',()=>checkFFT(runtime));
  await run('Original NVIDIA ocean stages match native cuFFT pipeline',()=>checkOcean(runtime));
  await run('Original NVIDIA optimized DCT matches native on padded and full images',()=>checkDctOptimized(runtime));
  await run('Original NVIDIA floating DCT stages and rounding match native CUDA',()=>checkDctFloat(runtime));
  await run('All original NVIDIA imageDenoising kernels match native portrait captures',()=>checkDenoising(runtime));
  await run('Original NVIDIA SobelShared matches native CUDA',()=>checkSobelShared(runtime));
  await run('Original NVIDIA Sobel image kernels match native CUDA',()=>checkSobelImages(runtime));
  await run('Short narrowing, promotion and launch values match native CUDA',()=>checkShortValues(runtime));
  await run('Original NVIDIA ComputeSobel matches native CUDA across five scales',()=>checkSobelNeighborhoods(runtime));
  await run('Packed byte stores preserve neighbouring pixels and match native CUDA',()=>checkByteStores(runtime));
  await run('Original RGBA box filter matches native row and column images',()=>checkBoxFilter(runtime));
  await run('Original sampled-volume marching cubes matches native CUDA at three isovalues',()=>checkMarchingVolume(runtime));
  await run('Shared pointer helpers match native CUDA with aliases and offsets',()=>checkSharedPointers(runtime));
  await run('Shared vector references match native CUDA after synchronization',()=>checkSharedVectorReferences(runtime));
  await run('Original NVIDIA marching-cubes classification matches native CUDA',()=>checkMarchingClassify(runtime));
  await run('Original NVIDIA interpolation through vector references matches native CUDA',()=>checkVectorReferences(runtime));
  await run('Vector launch arguments and packed byte reads match native CUDA',()=>checkVectorByteArguments(runtime));
  await run('Linear texture fetch matches native CUDA',()=>checkLinearTextures(runtime));
  await run('Original NVIDIA volume filter passes match native CUDA',()=>checkVolumeFilter(runtime));
  await run('sizeof and byte surface sampling match native CUDA',async()=>{const source=await(await fetch('/tests/sizeof-byte-surface.cu')).text(),sizes=await runtime.kernel(source,{entry:'sizeProbe',workgroupSize:[32]}),writer=await runtime.kernel(source,{entry:'byteWrite',workgroupSize:[4,4,2]}),reader=await runtime.kernel(source,{entry:'byteRead',workgroupSize:[4,4,2]}),values=new Uint32Array([0,1,0x10000000,0x7fffffff,0xffffffff]),input=runtime.createBuffer(values),output=runtime.createBuffer(new Uint32Array(256)),volume=runtime.createTexture3D(null,{width:8,height:8,depth:4,format:'rgba8unorm',storage:true});try{runtime.batch().dispatch(sizes.bind({input,output},{}),[1]).submit();const actual=await runtime.read(output,Uint32Array);for(let i=0;i<5;i++){const product=BigInt(values[i])*16n,expected=[Number(product&0xffffffffn),Number(product>0xffffffffn),12,24];for(let c=0;c<4;c++)if(actual[i*4+c]!==expected[c])throw Error('sizeof mismatch');}runtime.batch().dispatch(writer.bind({dst:volume},{}),[2,2,2]).dispatch(reader.bind({src:volume,output},{}),[2,2,2]).submit();const bytes=await runtime.read(output,Uint32Array);for(let i=0;i<256;i++)if(bytes[i]!==i)throw Error('Byte surface round trip mismatch '+i);return {sizeofResults:20,byteValues:256,nativeExact:true};}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);runtime.destroyTexture(volume);}});
  await run('Concurrent packed byte writers preserve neighbouring channels and match native CUDA',async()=>{const source=await(await fetch('/tests/packed-storage.cu')).text(),n=257,kernel=await runtime.kernel(source,{entry:'componentStores',workgroupSize:[64]}),copy=await runtime.kernel(source,{entry:'copyColours',workgroupSize:[64]}),out=runtime.createBuffer(new Uint32Array(n+16).fill(0xdeadbeef)),output=runtime.createBuffer(new Uint32Array(n+16).fill(0xdeadbeef));try{runtime.batch().dispatch(kernel.bind({out},{n}),[Math.ceil(n*4/64)]).dispatch(copy.bind({input:out,output},{n}),[Math.ceil(n/64)]).submit();for(const resource of [out,output]){const actual=await runtime.read(resource,Uint32Array);for(let i=0;i<n+16;i++){const expected=i<n?(5|((i+33)&255)<<8|1<<16|((128+i)&255)<<24)>>>0:0xdeadbeef;if(actual[i]!==expected)throw Error('Packed storage mismatch '+i);}}}finally{runtime.destroyBuffer(out);runtime.destroyBuffer(output);}});
  await run('Four native CUDA tiles preserve votes, shuffles and independent loop counts',async()=>{
    const source=await(await fetch('/tests/native-tiles.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[128]}),out=runtime.createBuffer(2560);
    try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const values=await runtime.read(out,Int32Array);for(let i=0;i<128;i++){const warp=Math.floor(i/32),lane=i%32,expected=[warp+1,warp*100+7,lane+1,Math.ceil((17+warp*25)/32),lane];if(values.slice(i*5,i*5+5).join(',')!==expected.join(','))throw Error('Native tile mismatch '+i+': '+values.slice(i*5,i*5+5));}const divergent=source.replace('for(int i=lane;tile.any(i<17+warp*25);i+=32)iterations++;','if(lane<16){for(int i=lane;tile.any(i<17+warp*25);i+=32)iterations++;}');let rejected=false;try{await runtime.kernel(divergent,{workgroupSize:[128]});}catch(e){rejected=/uniform control flow/.test(e.message);}if(!rejected)throw Error('Divergent vote-loop entry was accepted');return {tiles:4,lanes:128,checkedValues:640,fixedSubgroupSize:32,logicalThreadRemapping:true,divergentEntryRejected:true};}finally{runtime.destroyBuffer(out);}
  });
  await run('Native tile memory phases preserve shared snapshots and inactive lanes',async()=>{
    const source=await(await fetch('/tests/native-tile-phases.cu')).text(),out=runtime.createBuffer(512);let checkedValues=0;
    try{for(const tile of [0,2]){const kernel=await runtime.kernel(source.replace('warp==0','warp=='+tile),{workgroupSize:[128]});runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const values=await runtime.read(out,Int32Array);for(let i=0;i<128;i++){const index=i%16,row=Math.floor(index/4),expected=index+1+2*row*(row+1);if(values[i]!==expected)throw Error('Tile phase mismatch '+tile+':'+i+'='+values[i]+' expected '+expected);checkedValues++;}}return {checkedValues,selectedTiles:[0,2],sharedSnapshots:true};}finally{runtime.destroyBuffer(out);}
  });
  await run('Local CUDA array initializers preserve ordering and zero-filled elements',async()=>{
    const source=await(await fetch('/tests/local-array-init.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[32]}),out=runtime.createBuffer(1280);
    try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const values=await runtime.read(out,Int32Array);for(let i=0;i<32;i++)if(values.slice(i*10,i*10+10).join(',')!==[i,i+1,i+2,0,i+3,i+3,0,0,9,32].join(','))throw Error('Local array initializer mismatch '+i);return {lanes:32,checkedValues:320,nestedZeroFill:true,leftToRight:true};}finally{runtime.destroyBuffer(out);}
  });
  await run('Original NVIDIA quadtree root partitions all 1024 native points',()=>checkQuadtreeRoot(runtime));
  await run('Original Chrono neighbour cells and particle mapping support partial blocks',()=>checkChronoNeighbors(runtime));
  await run('Original Chrono scoped enums and boolean parameter records match native CUDA',()=>checkChronoTypes(runtime));
  await run('Original Chrono grid hashing matches native dam-break inputs and boundary cases',()=>checkChronoHash(runtime));
  await run('Original Chrono Adami wall pressure matches native CUDA',()=>checkChronoAdami(runtime));
  await run('CUDA byte-stride bool buffers and volatile flags match native',()=>checkBoolStorage(runtime));
  await run('C++ switch fall-through and nested control match native CUDA',()=>checkSwitch(runtime));
  await run('Original Chrono property reordering and diagnostics match native CUDA',()=>checkChronoReorder(runtime));
  await run('Chrono selected markers feed native-matched neighbour construction',()=>checkChronoSelected(runtime));
  await run('Chrono normalized activity compaction matches native CUDA',()=>checkChronoCompact(runtime));
  await run('Original Chrono activity selection matches native CUDA',()=>checkChronoActivity(runtime));
  await run('Chrono 64-bit counters and double launch times match native CUDA',()=>checkChronoCounterTime(runtime));
  await run('Original Chrono neighbour lists match native CUDA for fluid and boundary markers',()=>checkChronoSearch(runtime));
  await run('Original NVIDIA recursive quadtree matches every native node and output point',()=>checkQuadtreeRoot(runtime,{recursive:true}));
  await run('Recursive GPU queues reject overflow and unfinished generations',async()=>{
    const source=await(await fetch('/tests/recursive-launches.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[32],objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:1,maxGenerations:4},scheduleDeviceLaunches:true}),out=runtime.createBuffer(768),arena=runtime.createObjectArena();
    try{for(const limit of [4,1,0]){runtime.batch().clear(out).submit();await kernel.runQueued({out},{depth:0,limit,fanout:1},[1],{objectArena:arena});const values=await runtime.read(out,Uint32Array);for(let i=0;i<192;i++)if(values[i]!==((i/32|0)<=limit?(i/32|0)+1:0))throw Error('Recursive generation output mismatch');}
      for(const [limit,fanout,pattern] of [[5,1,/generation limit/],[1,2,/overflow/]]){let rejected=false;try{await kernel.runQueued({out},{depth:0,limit,fanout},[1],{objectArena:arena});}catch(e){if(!pattern.test(e.message))throw e;rejected=true;}if(!rejected)throw Error('Invalid recursive launch was accepted');}return {checkedValues:576,arenaReuse:true,overflowRejected:true,generationLimitRejected:true};
    }finally{arena.dispose();runtime.destroyBuffer(out);}
  });
  await run('Original quadtree storage references preserve node and nested field mutations',async()=>{
    const source=await(await fetch('/tests/quadtree-storage-references.cu')).text(),options={valueBuffers:['nodes'],workgroupSize:[32]},write=await runtime.kernel(source,{...options,entry:'write_nodes'}),read=await runtime.kernel(source,{...options,entry:'read_nodes'}),alias=await runtime.kernel(source,{...options,entry:'write_alias_nodes'}),nodes=runtime.createBuffer(1024),out=runtime.createBuffer(384);
    try{for(const kernel of [write,read,alias,read]){runtime.batch().dispatch(kernel.bind({nodes,out}),[1]).submit();const values=await runtime.read(out);for(let i=0;i<32;i++)if(values.slice(i*3,i*3+3).join(',')!==[i+2,i+3,7].join(','))throw Error('Storage record reference mismatch '+i);}return {nodes:32,checkedValues:384,bufferAliasOffsets:true,indexCapturedOnce:true,nestedMutationPersisted:true,originalClassBodies:true};}finally{runtime.destroyBuffer(nodes);runtime.destroyBuffer(out);}
  });
  await run('Queued offset pointers preserve disjoint child ranges and reject invalid offsets',async()=>{
    const source=await(await fetch('/tests/queued-offset.cu')).text(),options={objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:8},scheduleDeviceLaunches:true},kernel=await runtime.kernel(source,{...options,entry:'offset_parent',workgroupSize:[8]}),bad=await runtime.kernel(source,{...options,entry:'invalid_offset',workgroupSize:[1]}),arena=runtime.createObjectArena(),badArena=runtime.createObjectArena(),out=runtime.createBuffer(new Int32Array(40).fill(-99));
    try{await kernel.runQueued({out},{},[1],{objectArena:arena});const values=await runtime.read(out,Int32Array);for(let i=0;i<40;i++){const expected=i<2||i>=34?-99:Math.floor((i-2)/4)*10+(i-2)%4;if(values[i]!==expected)throw Error('Queued offset range mismatch '+i);}for(const offset of [-1,41]){let rejected=false;try{await bad.runQueued({out},{offset},[1],{objectArena:badArena});}catch(e){rejected=/overflow/.test(e.message);}if(!rejected)throw Error('Invalid offset was not rejected');}const after=await runtime.read(out,Int32Array);if(after.join(',')!==values.join(','))throw Error('Invalid launch modified storage');return {childLaunches:8,checkedValues:40,invalidOffsetsRejected:2,parentAliasMutationIsolated:true};}finally{runtime.destroyBuffer(out);arena.dispose();badArena.dispose();}
  });
  await run('Original Parameters pass by value through queued child launches',async()=>{
    const source=await(await fetch('/tests/queued-record.cu')).text(),kernel=await runtime.kernel(source,{entry:'record_parent',workgroupSize:[32],objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:32},scheduleDeviceLaunches:true}),arena=runtime.createObjectArena(),out=runtime.createBuffer(640);
    try{await kernel.runQueued({out},{'params.point_selector':0,'params.num_nodes_at_this_level':1,'params.depth':3,'params.max_depth':8,'params.min_points_per_node':16},[1],{objectArena:arena});const values=await runtime.read(out,Int32Array);for(let i=0;i<32;i++)if(values.slice(i*5,i*5+5).join(',')!==[1,4,4+i,8,16].join(','))throw Error('Queued record snapshot mismatch '+i);return {childLaunches:32,checkedFields:160,parentMutationIsolated:true,originalParameters:true};}finally{runtime.destroyBuffer(out);arena.dispose();}
  });
  await run('Queued children use CUDA launch shared-memory allocation',async()=>{
    const source=await(await fetch('/tests/queued-shared.cu')).text(),kernel=await runtime.kernel(source,{entry:'shared_parent',workgroupSize:[32],objectHeap:'persistent',deviceLaunchQueue:{maxLaunches:4},scheduleDeviceLaunches:true}),arena=runtime.createObjectArena(),out=runtime.createBuffer(256);
    try{await kernel.runQueued({out},{},[1],{objectArena:arena});const values=await runtime.read(out,Int32Array);for(let i=0;i<64;i++)if(values[i]!==Math.floor(i/32)*100+31-i%32)throw Error('Queued shared-memory mismatch');return {checkedValues:64,childBlocks:2,sharedBytesPerBlock:128};}finally{runtime.destroyBuffer(out);arena.dispose();}
  });
  await run('Volatile compound updates preserve integer and float shared values',async()=>{
    const source=await(await fetch('/tests/volatile-compound.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[32]}),out=runtime.createBuffer(256),fout=runtime.createBuffer(128);
    try{runtime.batch().dispatch(kernel.bind({out,fout}),[1]).submit();const ints=await runtime.read(out,Int32Array),floats=await runtime.read(fout);for(let i=0;i<32;i++){const lane=(i+1)%32;for(let row=0;row<2;row++)if(ints[i*2+row]!==(((row*100+lane+7)*3>>1)^5))throw Error('Volatile integer mismatch');if(floats[i]!==((lane+1)/2))throw Error('Volatile float mismatch');}return {checkedValues:96,separateLoadStore:true};}finally{runtime.destroyBuffer(out);runtime.destroyBuffer(fout);}
  });
  await run('Volatile shared pointer slots exchange values across GPU lanes',async()=>{
    const source=await(await fetch('/tests/volatile-pointer.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[32],sharedMemoryBytes:512}),out=runtime.createBuffer(512);
    try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const values=await runtime.read(out,Int32Array);for(let i=0;i<32;i++)for(let q=0;q<4;q++)if(values[i*4+q]!==q*100+(i+1)%32)throw Error('Volatile shared pointer mismatch');return {checkedValues:128,sharedPointerSlots:4};}finally{runtime.destroyBuffer(out);}
  });
  await run('Original quadtree Parameters constructors preserve const recursion limits',async()=>{
    const source=await(await fetch('/tests/quadtree-parameters.cu')).text(),kernel=await runtime.kernel(source,{entry:'parameter_values',workgroupSize:[32]}),out=runtime.createBuffer(768);
    try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const values=await runtime.read(out,Int32Array);for(let i=0;i<32;i++)if(values.slice(i*6,i*6+6).join(',')!==[0,1,0,16,i+5,16].join(','))throw Error('Parameters constructor changed parent or const fields');return {threads:32,constructors:96,originalParameters:true};}finally{runtime.destroyBuffer(out);}
  });
  await run('Original quadtree getters retain nested const field references',async()=>{
    const source=await(await fetch('/tests/quadtree-field-references.cu')).text(),kernel=await runtime.kernel(source,{entry:'field_references',workgroupSize:[32]}),out=runtime.createBuffer(384);
    try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const v=await runtime.read(out);for(let i=0;i<32;i++)if(v[i*3]!==1||v[i*3+1]!==i+2||v[i*3+2]!==i+3)throw Error('Field reference became a stale value copy');return {references:64,ownerMutationVisible:true,originalBoundingBoxAndNode:true};}finally{runtime.destroyBuffer(out);}
  });
  await run('Original Points retains captured scalar buffers across GPU submissions',()=>checkCapturedPoints(runtime));
  await run('Private class state and method access preserve CUDA values',async()=>{
    const source=await(await fetch('/tests/private-class.cu')).text(),kernel=await runtime.kernel(source,{entry:'private_class_values',workgroupSize:[32]}),out=runtime.createBuffer(128);
    try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const values=await runtime.read(out,Int32Array);for(let i=0;i<32;i++)if(values[i]!==3*i+7)throw Error('Private class result mismatch');return {values:32,privateFields:true,privateMethods:true,sameClassReceiver:true};}finally{runtime.destroyBuffer(out);}
  });
  await run('Complete original Bezier parent and indirect children match native',()=>checkBezierScheduled(runtime));
  await run('Original Bezier parent computes native counts and queues child work',()=>checkBezierParent(runtime));
  await run('Original Bezier child and cleanup match all native curves',()=>checkBezierChild(runtime));
  await run('Device allocations persist, match native CUDA and reuse freed slots',()=>checkDeviceHeap(runtime));
  await run('Bezier pointer records preserve opaque identity and clear independently',async()=>{
    const source=await(await fetch('/tests/bezier-pointer-record.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[64]}),stride=kernel.artifact.metadata.bindings.find(b=>b.name==='records').stride;
    if(stride!==32)throw Error('Unexpected WebGPU record stride');const words=new Uint32Array(256*8),floats=new Float32Array(words.buffer);for(let i=0;i<256;i++){for(let j=0;j<6;j++)floats[i*8+j]=(i*6+j)*.125;words[i*8+6]=[0,1,0x80000000,0xffffffff][i%4];words[i*8+7]=4+i%29;}
    const records=runtime.createBuffer(words),copies=runtime.createBuffer(words.byteLength),flags=runtime.createBuffer(256*4);
    try{runtime.batch().dispatch(kernel.bind({records,copies,flags}),[4]).submit();const copied=await runtime.read(copies,Uint32Array),cleared=await runtime.read(records,Uint32Array),seen=await runtime.read(flags,Uint32Array);for(let i=0;i<words.length;i++){if(copied[i]!==words[i]||cleared[i]!==(i%8===6?0:words[i]))throw Error('Record copy/clear mismatch '+i);}for(let i=0;i<256;i++)if(seen[i]!==Number(words[i*8+6]!==0))throw Error('Pointer identity mismatch');return {records:256,stride,opaquePointerBitsPreserved:true,independentClear:true,nativeAbi:false};}finally{for(const b of [records,copies,flags])runtime.destroyBuffer(b);}
  });
  await run('CUDA vector free operators dispatch their declared bodies',async()=>{const source=await(await fetch('/tests/vector-free-operators.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),out=runtime.createBuffer(32);try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const actual=await runtime.read(out),expected=[1.75,2.5,-3,-4,4,7,2,5];if(actual.some((v,i)=>v!==expected[i]))throw Error('Vector operator mismatch '+actual);return {values:8,originalNvidiaOperators:true,nonstandardOverloadBodyVerified:true};}finally{runtime.destroyBuffer(out);}});
  await run('Reference chains and sequenced expression updates match native CUDA',async()=>{const source=await(await fetch('/tests/reference-effects.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),out=runtime.createBuffer(52);try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();const actual=await runtime.read(out),expected=[1.25,3,3,1,3,1,1,3,12,4,4.125,Math.fround(.1),-0];if(actual.some((v,i)=>v!==expected[i]))throw Error('Reference expression mismatch '+Array.from(actual));return {valuesMatched:13,negativeZeroSignPreserved:Object.is(actual[12],-0),comparison:'Exact numeric values; zero sign recorded separately'};}finally{runtime.destroyBuffer(out);}});
  await run('Packed uchar4 and Boolean launch snapshots match native CUDA',async()=>{const source=await(await fetch('/tests/packed-launch.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),output=runtime.createBuffer(new Uint32Array(8).fill(0xdeadbeef));try{const inv=kernel.bind({output},{colour:0x80ff0100,choose:true,slot:0}),batch=runtime.batch(),colours=[0x80ff0100,0x12345678,0xffffffff,0],choices=[true,false,1,0];for(let slot=0;slot<4;slot++){inv.setScalars({colour:colours[slot],choose:choices[slot],slot});batch.dispatch(inv,[1]);}batch.submit();const actual=await runtime.read(output,Uint32Array),expected=[0x80ff0100,0x78563412,0xffffffff,0,0xdeadbeef,0xdeadbeef,0xdeadbeef,0xdeadbeef];if(actual.some((v,i)=>v!==expected[i]))throw Error('Packed launch snapshot mismatch');}finally{runtime.destroyBuffer(output);}});
  await run('Desktop conditional extraction matches both native CUDA branches',async()=>{const raw=await(await fetch('/tests/conditional-import.cu')).text(),source=kernelSource(raw).source;for(const PICK of [0,1]){const kernel=await runtime.kernel(source,{entry:'conditionalKernel',workgroupSize:[1],defines:{PICK}}),out=runtime.createBuffer(4);try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();if((await runtime.read(out,Int32Array))[0]!== (PICK?19:7))throw Error('Conditional branch mismatch');}finally{runtime.destroyBuffer(out);}}});
  await run('Original NVIDIA bilateral filter matches native images and independent reference',()=>checkBilateral(runtime));
  await run('Original NVIDIA postProcessGL RGBA texture and shared tile match native images',()=>checkPostProcess(runtime));
  await run('Indexed shared uchar4 macro and definition branches match native CUDA',async()=>{const source=await(await fetch('/tests/indexed-macros.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),output=runtime.createBuffer(4);try{runtime.batch().dispatch(kernel.bind({output}),[1]).submit();const actual=await runtime.read(output,Uint32Array);if(actual[0]!==0xff0d0b07)throw Error('Indexed macro value mismatch');}finally{runtime.destroyBuffer(output);}});
  await run('Original NVIDIA FDTD3d stencil uses constant coefficients across three volume steps',async()=>{
    const source=await(await fetch('/tests/fdtd3d-kernel.cuh')).text(),kernel=await runtime.kernel(source,{workgroupSize:[32,4,1]});
    for(const [dimx,dimy,dimz]of [[32,4,3],[35,7,5],[64,8,9],[32,16,16]]){
      let expected=fdtdInitial(dimx,dimy,dimz),input=runtime.createBuffer(expected),output=runtime.createBuffer(expected);
      try{const scalars={dimx,dimy,dimz,...Object.fromEntries(fdtdCoefficients.map((v,i)=>[`constant.stencil[${i}]`,v]))},inv=kernel.bind({input,output},scalars);
        for(let step=0;step<3;step++){expected=fdtdStep(expected,dimx,dimy,dimz);runtime.batch().dispatch(inv,[Math.ceil(dimx/32),Math.ceil(dimy/4),1]).copy(output,input).submit();const actual=await runtime.read(output);for(let i=0;i<actual.length;i++)if(!Number.isFinite(actual[i])||Math.abs(actual[i]-expected[i])>3e-5)throw Error(`FDTD mismatch ${dimx}x${dimy}x${dimz} step=${step} index=${i}`);}
      }finally{await runtime.idle();runtime.destroyBuffer(input);runtime.destroyBuffer(output);}
    }
  });
  await run('Pointer shifts preserve float4 stride, negative halo origins, aliases and by-value helper pointers',async()=>{
    const source='__device__ float4 read(const float4* p,int p_offset){p+=p_offset;return p[0];}__global__ void k(const float4* input,float4* out){const float4* saved=input;input-=2;const float4* alias=input;out[0]=read(alias,2);out[1]=saved[1];alias+=3;out[2]=alias[0];}';
    const input=runtime.createBuffer(Float32Array.from({length:16},(_,i)=>i)),out=runtime.createBuffer(new Float32Array(12));try{const kernel=await runtime.kernel(source,{workgroupSize:[1]});runtime.batch().dispatch(kernel.bind({input,out},{}),[1]).submit();const actual=await runtime.read(out);if(String(actual)!=='0,1,2,3,4,5,6,7,4,5,6,7')throw Error('Pointer stride, alias or helper mutation mismatch');}finally{await runtime.idle();runtime.destroyBuffer(input);runtime.destroyBuffer(out);}
  });
  await run('GPU range copy uses byte offsets and preserves surrounding values',async()=>{const a=runtime.createBuffer(new Float32Array([1,2,3,4])),b=runtime.createBuffer(new Float32Array([9,9,9,9,9,9]));try{runtime.batch().copy(a,b,{sourceOffset:4,targetOffset:8,byteLength:8}).submit();const actual=await runtime.read(b);if(actual.join(',')!=='9,9,2,3,9,9')throw Error('Range copy mismatch');}finally{runtime.destroyBuffer(a);runtime.destroyBuffer(b);}});
  await run('Original NVIDIA recursive Gaussian four-pass RGBA image filter',async()=>{const source=await(await fetch('/tests/recursive-gaussian.cuh')).text(),filter=await runtime.kernel(source,{entry:'d_recursiveGaussian_rgba',workgroupSize:[128,1,1]}),transpose=await runtime.kernel(source,{entry:'d_transpose',workgroupSize:[16,16,1]});for(const [w,h]of [[17,9],[32,16],[128,64]]){let expected=gaussianInitial(w,h),a=runtime.createBuffer(expected),b=runtime.createBuffer(new Uint32Array(expected.length).fill(0xdeadbeef));try{const params=gaussianCoefficients();for(const [width,height]of [[w,h],[h,w]]){expected=gaussianColumns(expected,width,height,params);runtime.batch().dispatch(filter.bind({id:a,od:b},{w:width,h:height,...params}),[Math.ceil(width/128)]).submit();let actual=await runtime.read(b,Uint32Array);for(let i=0;i<actual.length;i++)if(i<w*h?gaussianPixelError(actual[i],expected[i])>2:actual[i]!==expected[i])throw Error('Gaussian column mismatch '+i);expected=gaussianTranspose(expected,width,height);runtime.batch().dispatch(transpose.bind({idata:b,odata:a},{width,height}),[Math.ceil(width/16),Math.ceil(height/16)]).submit();actual=await runtime.read(a,Uint32Array);for(let i=0;i<actual.length;i++)if(i<w*h?gaussianPixelError(actual[i],expected[i])>2:actual[i]!==expected[i])throw Error('Gaussian transpose mismatch '+i);}}finally{runtime.destroyBuffer(a);runtime.destroyBuffer(b);}}});
  await run('Original NVIDIA simpleTexture3D samples real normalized 3D textures',async()=>{const source=await(await fetch('/tests/texture3d-kernel.cuh')).text(),kernel=await runtime.kernel(source,{entry:'d_render',workgroupSize:[8,8,1]}),data=texture3dInitial();for(const filter of ['linear','nearest']){const texture=runtime.createTexture3D(data,{width:8,height:4,depth:4,filter}),output=runtime.createBuffer(new Uint32Array(17*9+16).fill(0xdeadbeef));try{for(const w of [-.25,.13,.5,1.25]){runtime.batch().dispatch(kernel.bind({d_output:output,texObj:texture},{imageW:17,imageH:9,w}),[3,2,1]).submit();const actual=await runtime.read(output,Uint32Array);for(let i=0;i<actual.length;i++){const expected=i<17*9?Math.trunc(sampleTexture3D(data,[8,4,4],[(i%17)/17,Math.floor(i/17)/9,w],filter==='linear')):0xdeadbeef;if(Math.abs(actual[i]-expected)>(i<17*9?(filter==='linear'?2:1):0))throw Error('Texture mismatch '+filter+' w='+w+' pixel='+i+' actual='+actual[i]+' expected='+expected);}}}finally{runtime.destroyTexture(texture);runtime.destroyBuffer(output);}}});
  await run('Ray helper math uses runtime float vectors for normalization and dot products',async()=>{const source=await(await fetch('/tests/ray-vector-math.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),input=runtime.createBuffer(Float32Array.from([3,0,4,1])),output=runtime.createBuffer(64);try{const inv=kernel.bind({input,output});for(const [data,expected]of [[[3,0,4,1],[.6,0,.8,1,.5,0,.5,1,.6,.5,.8,2.4,1.1,1.5,3.3,1]],[[0,0,5,2],[0,0,1,1,0,0,.5,1,.5,.5,1,2,.5,1.5,3.5,1]]]){runtime.write(input,Float32Array.from(data));runtime.batch().dispatch(inv,[1]).submit();const actual=await runtime.read(output);for(let i=0;i<16;i++)if(!Number.isFinite(actual[i])||Math.abs(actual[i]-expected[i])>2e-6)throw Error('Ray math mismatch '+i);}}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('Local structs and constant camera matrices retain value semantics on GPU',async()=>{const source=await(await fetch('/tests/local-structs.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),input=runtime.createBuffer(Float32Array.from([1,2,3,1,4,5,6,1])),output=runtime.createBuffer(64);try{const scalars={row:0};for(let row=0;row<3;row++)for(let j=0;j<4;j++)scalars['constant.camera.m['+row+'].'+'xyzw'[j]]=row*4+j+1;const inv=kernel.bind({input,output},scalars);for(let row=0;row<3;row++){inv.setScalars({row});runtime.batch().dispatch(inv,[1]).submit();const actual=await runtime.read(output),expected=[5,7,9,1,1,2,3,1,...(row===0?[99,2,3,4]:[row*4+1,row*4+2,row*4+3,row*4+4]),1,2,3,4];for(let i=0;i<16;i++)if(actual[i]!==expected[i])throw Error('Struct mismatch row='+row+' component='+i);}}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('Paired 3D volume and 1D float4 transfer textures sample on NVIDIA hardware',async()=>{const source=await(await fetch('/tests/transfer-texture.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[8]}),coordinates=runtime.createBuffer(Float32Array.from(transferCoordinates)),output=runtime.createBuffer(transferCoordinates.length*16),volume=runtime.createTexture3D(new Uint8Array(8).fill(128),{width:2,height:2,depth:2});try{for(const filter of ['linear','nearest']){const transfer=runtime.createTexture1D(Float32Array.from(transferValues),{filter});try{for(const shift of [0,.25]){runtime.batch().dispatch(kernel.bind({coordinates,output,volume,transfer},{shift,n:transferCoordinates.length}),[1]).submit();const actual=await runtime.read(output);for(let i=0;i<transferCoordinates.length;i++){const expected=transferSample(transferValues,transferCoordinates[i]+128/255*shift,filter==='linear');for(let c=0;c<4;c++)if(!Number.isFinite(actual[i*4+c])||Math.abs(actual[i*4+c]-expected[c])>.005)throw Error('Transfer texture mismatch '+i+' channel='+c);}}}finally{runtime.destroyTexture(transfer);}}}finally{runtime.destroyBuffer(coordinates);runtime.destroyBuffer(output);runtime.destroyTexture(volume);}});
  await run('Original NVIDIA volume matrix overloads and const references',async()=>{const source=await(await fetch('/tests/volume-mul.cuh')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),input=runtime.createBuffer(Float32Array.from([1,2,3,1])),output=runtime.createBuffer(32),scalars={};[2,0,0,10,0,3,0,20,0,0,4,30].forEach((v,i)=>scalars['constant.c_invViewMatrix.m['+Math.floor(i/4)+'].'+'xyzw'[i%4]]=v);try{runtime.batch().dispatch(kernel.bind({input,output},scalars),[1]).submit();const actual=await runtime.read(output),expected=[2,6,12,0,12,26,42,1];for(let i=0;i<8;i++)if(actual[i]!==expected[i])throw Error('Matrix overload mismatch '+i);}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('Expression macros preserve signed IMAD and argument precedence on GPU',async()=>{const source=await(await fetch('/tests/expression-macros.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),input=runtime.createBuffer(Int32Array.from([16777217,-8388607,2,3])),output=runtime.createBuffer(20);try{runtime.batch().dispatch(kernel.bind({input,output}),[1]).submit();const actual=await runtime.read(output,Int32Array),expected=[5,-16777213,11,25,17];for(let i=0;i<5;i++)if(actual[i]!==expected[i])throw Error('Expression macro mismatch '+i);}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('Complete original convolutionTexture matches native CUDA and independent reference',()=>checkConvolutionTexture(runtime));
  await run('Texture helpers forward distinct images and samplers through integer specializations',async()=>{const source=await(await fetch('/tests/texture-helpers.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[4]}),first=runtime.createTexture2D(Float32Array.from([1,2,3,4]),{width:4,height:1,filter:'nearest',addressMode:'clamp-to-edge'}),second=runtime.createTexture2D(Float32Array.from([10,20,30,40]),{width:4,height:1,filter:'linear',addressMode:'clamp-to-edge'}),output=runtime.createBuffer(32);try{runtime.batch().dispatch(kernel.bind({output,first,second}),[1]).submit();const actual=await runtime.read(output),expected=[210,370,580,790,120,215,335,455];for(let i=0;i<8;i++)if(actual[i]!==expected[i])throw Error('Texture helper mismatch '+i);}finally{runtime.destroyTexture(first);runtime.destroyTexture(second);runtime.destroyBuffer(output);}});
  await run('Volume and transfer textures pass through helper parameters',async()=>{const kernel=await runtime.kernel('__device__ float v(cudaTextureObject_t t){return tex3D<float>(t,0.5f,0.5f,0.5f);}__device__ float4 f(cudaTextureObject_t t){return tex1D<float4>(t,0.5f);}__global__ void k(float4* out,cudaTextureObject_t volume,cudaTextureObject_t table){out[0]=f(table)*v(volume);}',{workgroupSize:[1]}),volume=runtime.createTexture3D(new Uint8Array([128]),{width:1,height:1,depth:1}),table=runtime.createTexture1D(Float32Array.from([.25,.5,.75,1])),out=runtime.createBuffer(16);try{runtime.batch().dispatch(kernel.bind({out,volume,table}),[1]).submit();const actual=await runtime.read(out);for(let i=0;i<4;i++)if(!Number.isFinite(actual[i])||Math.abs(actual[i]-(i+1)/4*128/255)>1e-6)throw Error('Helper texture format mismatch');}finally{runtime.destroyTexture(volume);runtime.destroyTexture(table);runtime.destroyBuffer(out);}});
  await run('Unused 2D texture helper retains a valid shader module',async()=>{const kernel=await runtime.kernel('__device__ float unused(cudaTextureObject_t tex){return tex2D<float>(tex,.5f,.5f);}__global__ void k(int* out){out[0]=7;}',{workgroupSize:[1]}),out=runtime.createBuffer(4);try{runtime.batch().dispatch(kernel.bind({out}),[1]).submit();if((await runtime.read(out,Int32Array))[0]!==7)throw Error('Unused texture helper shader failed');}finally{runtime.destroyBuffer(out);}});
  await run('Complete original bicubicTexture modes match native CUDA and independent reference',()=>checkBicubic(runtime));
  await run('Packed uchar4 matches native CUDA byte layout and promotions',async()=>{const n=257,source=await(await fetch('/tests/packed-uchar4.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[64]}),values=runtime.createBuffer(Float32Array.from({length:n},(_,i)=>(i*37)%256+.75)),output=runtime.createBuffer(new Uint32Array(n*2+16).fill(0xdeadbeef)),checks=runtime.createBuffer(n*4);try{runtime.batch().dispatch(kernel.bind({values,output,checks},{n}),[5]).submit();const actual=await runtime.read(output,Uint32Array),diagnostic=await runtime.read(checks,Int32Array);for(let i=0;i<n;i++){const x=(i*37)%256,y=(i-130)&255,z=(i*17)&255,p=(x|(y<<8)|(z<<16)|(255<<24))>>>0,q=(((x+250)&255)|(((y-1)&255)<<8)|(((z+1)&255)<<16))>>>0;if(actual[i*2]!==p||actual[i*2+1]!==q||diagnostic[i]!==-1700+x+4)throw Error('Packed byte mismatch '+i);}if(actual.slice(n*2).some(x=>x!==0xdeadbeef))throw Error('Packed output guard changed');}finally{runtime.destroyBuffer(values);runtime.destroyBuffer(output);runtime.destroyBuffer(checks);}});
  await run('Helper default arguments match native CUDA including templates and textures',async()=>{const source=await(await fetch('/tests/helper-defaults.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),input=runtime.createBuffer(Float32Array.from([2.5,-3.5])),output=runtime.createBuffer(40),tex=runtime.createTexture2D(Float32Array.from([1,2,3,4,5,6,7,8]),{width:4,height:2,normalizedCoords:false,addressMode:'clamp-to-edge'});try{runtime.batch().dispatch(kernel.bind({input,output,tex}),[1]).submit();const actual=await runtime.read(output),expected=[2,4.5,12.5,6.5,-2,2,5,-10,9,3];for(let i=0;i<10;i++)if(actual[i]!==expected[i])throw Error('Helper default mismatch '+i);}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);runtime.destroyTexture(tex);}});
  await run('Multiple helper template types match native CUDA including texture forwarding',async()=>{const source=await(await fetch('/tests/multiple-helper-types.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),input=runtime.createBuffer(Float32Array.from([2.5,-3.5])),output=runtime.createBuffer(24),tex=runtime.createTexture2D(Float32Array.from([1,2,3,4,5,6,7,8]),{width:4,height:2,normalizedCoords:false,addressMode:'clamp-to-edge'});try{runtime.batch().dispatch(kernel.bind({input,output,tex}),[1]).submit();const actual=await runtime.read(output),expected=[3,-30,-27,7.5,2,4.5];for(let i=0;i<6;i++)if(actual[i]!==expected[i])throw Error('Multiple helper types mismatch '+i);}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);runtime.destroyTexture(tex);}});
  await run('Integer helper expressions terminate at negative specializations on GPU',async()=>{const source=await(await fetch('/tests/integer-helper-expressions.cu')).text(),kernel=await runtime.kernel(source,{workgroupSize:[1]}),input=runtime.createBuffer(Int32Array.from([3,-2])),output=runtime.createBuffer(24);try{runtime.batch().dispatch(kernel.bind({input,output}),[1]).submit();const actual=await runtime.read(output,Int32Array),expected=[187,0,7,-2,-2147483648,-3];for(let i=0;i<6;i++)if(actual[i]!==expected[i])throw Error('Integer helper expression mismatch '+i);}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);}});
  await run('Original NVIDIA surface write and texture pipeline matches native CUDA',()=>checkSurfaceWrite(runtime));
  await run('Original NVIDIA 2D texture rotation matches native CUDA images',()=>checkTexture2D(runtime));
  await run('Complete original NVIDIA volume renderer matches native CUDA images',async()=>{const source=await(await fetch('/tests/volume-render.cuh')).text(),kernel=await runtime.kernel(source,{entry:'d_render',workgroupSize:[8,8,1]}),volume=runtime.createTexture3D(new Uint8Array(await(await fetch('/showcases/volume-render/Bucky.raw')).arrayBuffer()),{width:32,height:32,depth:32,addressMode:'clamp-to-edge'}),table=runtime.createTexture1D(Float32Array.from(volumeTransfer)),comparisons=[];try{for(let scenario=0;scenario<3;scenario++){const {width,height,scalars}=volumeCase(scenario),initial=new Uint32Array(width*height+16).fill(0xdeadbeef);initial.fill(0,0,width*height);const output=runtime.createBuffer(initial);try{runtime.batch().dispatch(kernel.bind({d_output:output,tex:volume,transferTex:table},scalars),[Math.ceil(width/8),Math.ceil(height/8),1]).submit();const actual=await runtime.read(output,Uint32Array),expected=new Uint32Array(await(await fetch('/reports/volume-render-native-'+scenario+'.bin')).arrayBuffer());if(expected.length!==initial.length)throw Error('Native volume image size mismatch');let maxChannelError=0,differentPixels=0;for(let i=0;i<actual.length;i++){if(i>=width*height){if(actual[i]!==0xdeadbeef)throw Error('Volume output guard changed');continue;}if(actual[i]!==expected[i])differentPixels++;for(let c=0;c<4;c++){const error=Math.abs(((actual[i]>>>(c*8))&255)-((expected[i]>>>(c*8))&255));maxChannelError=Math.max(maxChannelError,error);if(error>2)throw Error('Volume render mismatch scenario='+scenario+' pixel='+i);}}comparisons.push({scenario,width,height,maxChannelError,differentPixels});}finally{runtime.destroyBuffer(output);}}return {comparisons};}finally{runtime.destroyTexture(volume);runtime.destroyTexture(table);}});
  await run('Original NVIDIA Haar full transform with GPU coefficient range copies',async()=>{
    const source=await(await fetch('/tests/haar-kernel.cuh')).text();for(const n of [4,32,1024,4096]){const initial=haarInitial(n),expected=haarReference(initial),padded=new Float32Array(n+16).fill(-12345);padded.set(initial);const id=runtime.createBuffer(padded),od=runtime.createBuffer(new Float32Array(n+16).fill(-12345)),approx_final=runtime.createBuffer(new Float32Array(n));try{const batch=runtime.batch();for(let length=n;length>1;){const block=Math.min(512,length/2),groups=length/(2*block),kernel=await runtime.kernel(source,{entry:'dwtHaar1D',workgroupSize:[block,1,1],sharedMemoryBytes:(2*block+Math.floor(2*block/16))*4});batch.dispatch(kernel.bind({id,od,approx_final},{dlevels:Math.log2(2*block),slength_step_half:length/2,bdim:block}),[groups]).copy(approx_final,groups===1?od:id,{byteLength:groups*4});length=groups;}batch.submit();const actual=await runtime.read(od);for(let i=0;i<actual.length;i++)if(!Number.isFinite(actual[i])||Math.abs(actual[i]-(i<n?expected[i]:-12345))>3e-5)throw Error('Haar mismatch n='+n+' index='+i);}finally{runtime.destroyBuffer(id);runtime.destroyBuffer(od);runtime.destroyBuffer(approx_final);}}
  });
  await run('Original NVIDIA separable convolution runs both passes with mutable pointer offsets',async()=>{
    const source=await(await fetch('/tests/convolution-separable.cuh')).text(),rows=await runtime.kernel(source,{entry:'convolutionRowsKernel',workgroupSize:[16,4,1]}),columns=await runtime.kernel(source,{entry:'convolutionColumnsKernel',workgroupSize:[16,8,1]});
    for(const [imageW,imageH,pitch]of [[128,64,128],[256,128,272]]){let expected=separableInitial(imageW,imageH,pitch),a=runtime.createBuffer(expected),b=runtime.createBuffer(new Float32Array(expected.length).fill(-12345));try{
      const scalars={imageW,imageH,pitch,...Object.fromEntries(separableCoefficients.map((v,i)=>[`constant.c_Kernel[${i}]`,v]))};
      for(const [kernel,axis,groups]of [[rows,'rows',[imageW/128,imageH/4]],[columns,'columns',[imageW/16,imageH/64]]]){expected=separableReference(expected,imageW,imageH,pitch,axis);runtime.batch().dispatch(kernel.bind({d_Dst:b,d_Src:a},scalars),groups).submit();const actual=await runtime.read(b);for(let i=0;i<actual.length;i++)if(!Number.isFinite(actual[i])||Math.abs(actual[i]-expected[i])>3e-5)throw Error('Separable convolution mismatch '+axis+' '+i);[a,b]=[b,a];}
    }finally{await runtime.idle();runtime.destroyBuffer(a);runtime.destroyBuffer(b);}}
  });
  await run('GPU rejects divergent entry into a helper barrier',async()=>{
    try{await runtime.kernel('__device__ void barrier(){__syncthreads();} __global__ void k(){if(threadIdx.x==0u)barrier();}',{workgroupSize:[4,1,1]});}catch(error){if(/uniform/i.test(error.message))return;throw error;}
    throw Error('Divergent helper barrier was accepted');
  });
  // Explicitly exercise workgroup variants used by the tuner, beyond the catalogue defaults.
  for(const block of [64,256])await run(`SAXPY workgroup specialization ${block}`,async()=>{
    const n=1031,xData=Float32Array.from({length:n},(_,i)=>i*0.125),x=runtime.createBuffer(xData),y=runtime.createBuffer(n*4);
    try{const kernel=await runtime.kernel(sources.saxpy,{entry:'saxpy',workgroupSize:[block,1,1]});runtime.batch().dispatch(kernel.bind({x,y},{a:2,n}),[Math.ceil(n/block)]).submit();const out=await runtime.read(y);for(let i=0;i<n;i++)if(out[i]!==xData[i]*2)throw new Error(`Mismatch at ${i}`);}finally{runtime.destroyBuffer(x);runtime.destroyBuffer(y);}
  });
  return {schema:'cuda-webshader.correctness.v1',mode:'REAL WebGPU execution',device:runtime.describe(),date:new Date().toISOString(),total:results.length,passed:results.filter(r=>r.pass).length,failed:results.filter(r=>!r.pass).length,wallMs:performance.now()-start,results};
}
