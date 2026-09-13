import {FLOAT64_WGSL} from '../src/compiler/float64.js';
export async function checkDoubleMath(runtime){
 const input=new Uint32Array(await(await fetch(new URL('../reports/double-math-input.bin',import.meta.url))).arrayBuffer()),expected=new Uint32Array(await(await fetch(new URL('../reports/double-math-native.bin',import.meta.url))).arrayBuffer()),count=input.length/4;
 const source=FLOAT64_WGSL+`
 @group(0) @binding(0) var<storage,read> input: array<vec2<u32>>;
 @group(0) @binding(1) var<storage,read_write> output: array<vec2<u32>>;
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id: vec3<u32>){
  let i=id.x; if(i >= ${count}u){return;} let a=input[i*2u];let b=input[i*2u+1u];
  output[i*3u]=cw_d_sqrt(a);output[i*3u+1u]=cw_d_fmin(a,b);output[i*3u+2u]=cw_d_fmax(a,b);
 }`;
 const device=runtime.device;device.pushErrorScope('validation');
 const shader=device.createShaderModule({code:source}),pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}});
 const a=runtime.createBuffer(input),b=runtime.createBuffer(expected.byteLength);
 try{
  const bindings=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a.gpuBuffer}},{binding:1,resource:{buffer:b.gpuBuffer}}]});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(Math.ceil(count/128));pass.end();device.queue.submit([encoder.finish()]);
  const actual=await runtime.read(b,Uint32Array),error=await device.popErrorScope();if(error)throw error;
  const nan64=(data,i)=>(data[i+1]&0x7fffffff)>0x7ff00000||((data[i+1]&0x7fffffff)===0x7ff00000&&data[i]!==0),nan32=bits=>(bits&0x7fffffff)>0x7f800000;
  for(let i=0;i<actual.length;i+=2){if(nan64(expected,i)&&nan64(actual,i))continue;for(let c=0;c<2;c++)if(actual[i+c]!==expected[i+c])throw Error('Double math mismatch pair='+Math.floor(i/6)+' op='+((i/2)%3)+' expected='+expected[i+c].toString(16)+' actual='+actual[i+c].toString(16));}
  const compiled=await checkDoubleLocals(runtime);
  const integrationSource=await(await fetch(new URL('chrono-integration.cu',import.meta.url))).text();
  await runtime.kernel(integrationSource,{entry:'EulerStep_D',defines:{__CUDA_ARCH__:1},workgroupSize:[128]});
  return {pairs:count,results:count*3,bitExactExceptNaNPayloads:true,compiled,integrationKernelCompiled:true,integrationKernelExecuted:false};
 }finally{runtime.destroyBuffer(a);runtime.destroyBuffer(b);}
}

export async function checkDoubleLocals(runtime){
 const load=path=>fetch(new URL(path,import.meta.url)),source=await(await load('double-locals.cu')).text(),reference=new Uint8Array(await(await load('../reports/double-locals-native.bin')).arrayBuffer()),n=129;
 const input=Float32Array.from({length:n},(_,i)=>(i+1)*.125),records=Float32Array.from({length:n*2},(_,i)=>i%2?-Math.floor(i/2):i/2);input[n-1]=1e20;
 const math=await runtime.kernel(source,{entry:'doubleLocals',workgroupSize:[128]}),refs=await runtime.kernel(source,{entry:'recordRefs',workgroupSize:[128]}),a=runtime.createBuffer(input),b=runtime.createBuffer(n*12),p=runtime.createBuffer(records);
 try{runtime.batch().dispatch(math.bind({input:a,output:b},{n}),[2]).dispatch(refs.bind({pairs:p},{n}),[2]).submit();
 const out=await runtime.read(b,Uint32Array),pairs=await runtime.read(p,Uint32Array),actual=new Uint8Array(n*20);actual.set(new Uint8Array(out.buffer));actual.set(new Uint8Array(pairs.buffer),n*12);
 if(actual.length!==reference.length||actual.some((v,i)=>v!==reference[i]))throw Error('Compiled double locals or record scalar reference mismatch');
 return {bytesCompared:actual.length,nativeExact:true,quadraticCases:n,recordCases:n};
 }finally{runtime.destroyBuffer(a);runtime.destroyBuffer(b);runtime.destroyBuffer(p);}
}
