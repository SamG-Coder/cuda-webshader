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
