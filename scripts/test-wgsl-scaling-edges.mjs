import {createStaticServer} from './serve.mjs';import {chromium} from 'playwright';import {writeFileSync} from 'node:fs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const result=await page.evaluate(async()=>{
  const {GpuRuntime}=await import('/src/runtime/runtime.js'),{compile}=await import('/src/compiler/compiler.js'),{FLOAT64_WGSL}=await import('/src/compiler/float64.js'),{optimizeFloatScaling}=await import('/tests/experiments/wgsl-float-scaling.js');
  const adapter=await navigator.gpu.requestAdapter();if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA required');
  const errors=[],runtime=await GpuRuntime.create({onError:e=>errors.push(e.message)});
  try{
   const bits=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001];let seed=0xfeedface;for(let i=0;i<4096;i++){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;bits.push(seed>>>0);}
   const words=new Uint32Array(bits),values=new Float32Array(words.buffer),shifts=[-512,-149,-1,0,1,24,127,512],n=words.length;
   const artifact=compile('__global__ void scale(const float* input,unsigned int* output){output[0]=(unsigned int)input[0];}',{workgroupSize:[128]});
   artifact.wgsl=FLOAT64_WGSL+'\n@group(0) @binding(0) var<storage,read> input:array<f32>; @group(0) @binding(1) var<storage,read_write> output:array<u32>; @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>='+n+'u){return;}'+shifts.map((shift,j)=>`let d${j}=cw_d_mul(cw_d_from_f32(input[i]),vec2<u32>(0u,${(1023+shift)*1048576}u));output[i*16u+${2*j}u]=d${j}.x;output[i*16u+${2*j+1}u]=d${j}.y;`).join('\n')+'}';
   const optimized=optimizeFloatScaling(artifact),input=runtime.createBuffer(words),outputs=[];
   for(const a of [artifact,optimized.artifact]){const kernel=await runtime.kernel(a),output=runtime.createBuffer(n*16*4);runtime.batch().dispatch(kernel.bind({input,output},{}),[Math.ceil(n/128)]).submit();outputs.push(await runtime.read(output,Uint32Array));runtime.destroyBuffer(output);}
   if(optimized.replacements!==8)throw Error('Expected eight scale rewrites');for(let i=0;i<outputs[0].length;i++)if(outputs[0][i]!==outputs[1][i])throw Error('Scale bit mismatch at '+i);
   const bytes=new ArrayBuffer(8),view=new DataView(bytes);for(let i=0;i<n;i++)for(let j=0;j<shifts.length;j++){const expected=values[i]*2**shifts[j];if(Number.isNaN(expected))continue;view.setFloat64(0,expected,true);if(outputs[1][i*16+j*2]!==view.getUint32(0,true)||outputs[1][i*16+j*2+1]!==view.getUint32(4,true))throw Error('Independent scale mismatch at '+i+'/'+j);}
   if(errors.length)throw Error(errors.join('\n'));runtime.destroyBuffer(input);return {pass:true,values:n,operations:n*8,shifts,exactGpuBits:true,independentReference:true,softwareAdapterRequested:false};
  }finally{runtime.dispose();}
 });writeFileSync('.local/wgsl-scaling-edges.json',JSON.stringify(result,null,2));console.log(result);
}finally{await browser.close();await new Promise(r=>server.close(r));}
