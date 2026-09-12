import {FLOAT64_WGSL} from '../src/compiler/float64.js';
export async function checkFloat64(runtime){
 const input=new Uint32Array(await(await fetch('/reports/float64-input.bin')).arrayBuffer()),expected=new Uint32Array(await(await fetch('/reports/float64-native.bin')).arrayBuffer()),count=input.length/4;
 const source=FLOAT64_WGSL+`
 @group(0) @binding(0) var<storage,read> input: array<vec2<u32>>;
 @group(0) @binding(1) var<storage,read_write> output: array<vec2<u32>>;
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id: vec3<u32>){
  let i=id.x; if(i >= ${count}u){return;} let a=input[i*2u];let b=input[i*2u+1u];
  output[i*10u]=cw_d_add(a,b);output[i*10u+1u]=cw_d_sub(a,b);output[i*10u+2u]=cw_d_mul(a,b);output[i*10u+3u]=cw_d_div(a,b);
  output[i*10u+4u]=vec2<u32>(select(0u,1u,cw_d_lt(a,b))|select(0u,2u,cw_d_eq(a,b))|select(0u,4u,cw_d_lt(b,a)),0u);
  output[i*10u+5u]=vec2<u32>(bitcast<u32>(cw_d_to_f32(a)),bitcast<u32>(cw_d_to_f32(b)));
  output[i*10u+6u]=cw_d_from_f32(bitcast<f32>(a.x));output[i*10u+7u]=cw_d_from_u32(a.x);output[i*10u+8u]=cw_d_from_i32(bitcast<i32>(a.x));
  output[i*10u+9u]=vec2<u32>(bitcast<u32>(cw_d_u64_to_f32(a)),bitcast<u32>(cw_d_u64_to_f32(b)));
 }`;
 const device=runtime.device;device.pushErrorScope('validation');
 const shader=device.createShaderModule({code:source}),pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shader,entryPoint:'main'}});
 const a=runtime.createBuffer(input),b=runtime.createBuffer(expected.byteLength);
 try{
  const bindings=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a.gpuBuffer}},{binding:1,resource:{buffer:b.gpuBuffer}}]});
  const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(Math.ceil(count/128));pass.end();device.queue.submit([encoder.finish()]);
  const actual=await runtime.read(b,Uint32Array),error=await device.popErrorScope();if(error)throw error;
  const nan64=(data,i)=>(data[i+1]&0x7fffffff)>0x7ff00000||((data[i+1]&0x7fffffff)===0x7ff00000&&data[i]!==0),nan32=bits=>(bits&0x7fffffff)>0x7f800000;
  for(let i=0;i<actual.length;i+=2){const operation=(i/2)%10;if([0,1,2,3,6,7,8].includes(operation)&&nan64(expected,i)&&nan64(actual,i))continue;for(let c=0;c<2;c++){if(operation===5&&nan32(expected[i+c])&&nan32(actual[i+c]))continue;if(actual[i+c]!==expected[i+c])throw Error('Binary64 native mismatch pair='+Math.floor(i/20)+' operation='+operation+' word='+c+' expected='+expected[i+c].toString(16)+' actual='+actual[i+c].toString(16));}}
  const cudaSource=await(await fetch('/tests/float64-expression-kernel.cuh')).text(),kernel=await runtime.kernel(cudaSource,{entry:'expressions',workgroupSize:[128,1,1]});
  const aWords=Uint32Array.from({length:count},(_,i)=>input[i*4]),bWords=Uint32Array.from({length:count},(_,i)=>input[i*4+2]);
  const ca=runtime.createBuffer(aWords),cb=runtime.createBuffer(bWords),co=runtime.createBuffer(count*24);
  try{runtime.batch().dispatch(kernel.bind({a:ca,b:cb,out:co},{count}),[Math.ceil(count/128),1,1]).submit();const values=await runtime.read(co,Uint32Array),reference=new Uint32Array(await(await fetch('/reports/float64-expressions-native.bin')).arrayBuffer());if(reference.length!==values.length)throw Error('Wrong native expression capture length');for(let i=0;i<values.length;i++)if(values[i]!==reference[i]&&!(nan32(values[i])&&nan32(reference[i])))throw Error('Compiled double expression mismatch '+i+' expected='+reference[i].toString(16)+' actual='+values[i].toString(16));}finally{runtime.destroyBuffer(ca);runtime.destroyBuffer(cb);runtime.destroyBuffer(co);}
  return {pairs:count,results:count*10,compiledExpressions:count*6,bitExactExceptNaNPayloads:true,softwareAdapterRequested:false};
 }finally{runtime.destroyBuffer(a);runtime.destroyBuffer(b);}
}
