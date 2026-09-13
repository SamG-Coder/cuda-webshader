import {createStaticServer} from './serve.mjs';
import {chromium} from 'playwright';
import {writeFileSync} from 'node:fs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try {
 const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const result=await page.evaluate(async()=>{
  const {GpuRuntime}=await import('/src/runtime/runtime.js'),{compile}=await import('/src/compiler/compiler.js'),{FLOAT64_WGSL}=await import('/src/compiler/float64.js'),{optimizeLimbMultiply}=await import('/tests/experiments/wgsl-limb-multiply.js');
  const adapter=await navigator.gpu.requestAdapter();if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA required');
  const errors=[],runtime=await GpuRuntime.create({onError:e=>errors.push(e.message)});
  try {
   const captured=new Uint32Array(await(await fetch('/reports/float64-input.bin')).arrayBuffer());
   const input=new Uint32Array(captured.length+65536*4);input.set(captured);let seed=0x892174a;
   for(let i=captured.length;i<input.length;i++){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;input[i]=seed>>>0;}
   const n=input.length/4,artifact=compile('__global__ void multiply(const unsigned int* input,unsigned int* output){output[0]=input[0];}',{workgroupSize:[128]});
   artifact.wgsl=FLOAT64_WGSL+`\n@group(0) @binding(0) var<storage,read> input:array<u32>; @group(0) @binding(1) var<storage,read_write> output:array<u32>; @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=${n}u){return;} let d=cw_d_mul(vec2<u32>(input[4u*i],input[4u*i+1u]),vec2<u32>(input[4u*i+2u],input[4u*i+3u]));output[2u*i]=d.x;output[2u*i+1u]=d.y;}`;
   const optimized=optimizeLimbMultiply(artifact);if(optimized.replacements!==1)throw Error('Multiply not replaced');
   const a=runtime.createBuffer(input),results=[];
   for(const shader of [artifact,optimized.artifact]){const k=await runtime.kernel(shader),b=runtime.createBuffer(n*8);runtime.batch().dispatch(k.bind({input:a,output:b},{}),[Math.ceil(n/128)]).submit();results.push(await runtime.read(b,Uint32Array));runtime.destroyBuffer(b);}
   for(let i=0;i<results[0].length;i++)if(results[0][i]!==results[1][i])throw Error('GPU bit mismatch '+i);
   const view=new DataView(input.buffer),expected=new DataView(new ArrayBuffer(8));let finiteChecks=0;
   for(let i=0;i<n;i++){const value=view.getFloat64(i*16,true)*view.getFloat64(i*16+8,true);if(Number.isNaN(value))continue;expected.setFloat64(0,value,true);for(let j=0;j<2;j++)if(results[1][2*i+j]!==expected.getUint32(4*j,true))throw Error('Independent reference mismatch '+i);finiteChecks++;}
   runtime.destroyBuffer(a);if(errors.length)throw Error(errors.join('\n'));return {pass:true,pairs:n,independentNonNaNChecks:finiteChecks,exactGpuBitsIncludingNaNs:true,softwareAdapterRequested:false};
  }finally{runtime.dispose();}
 });writeFileSync('reports/wgsl-limb-multiply-edges.json',JSON.stringify(result,null,2));console.log(result);
}finally{await browser.close();await new Promise(r=>server.close(r));}
