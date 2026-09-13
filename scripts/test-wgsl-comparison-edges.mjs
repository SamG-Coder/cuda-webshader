import {createStaticServer} from './serve.mjs';
import {chromium} from 'playwright';
import {writeFileSync} from 'node:fs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const result=await page.evaluate(async()=>{
  const {GpuRuntime}=await import('/src/runtime/runtime.js'),{compile}=await import('/src/compiler/compiler.js'),{optimizeFloatComparisons}=await import('/tests/experiments/wgsl-float-comparisons.js');
  const adapter=await navigator.gpu.requestAdapter();if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA required');
  const errors=[],runtime=await GpuRuntime.create({onError:e=>errors.push(e.message)});
  try{
   const source=`__global__ void compare(const float* input,unsigned int* output,unsigned int n){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double a=(double)input[i];double b=(double)input[(i+1)%n];output[6*i]=(double)input[i]<(double)input[(i+1)%n];output[6*i+1]=(double)input[i]==(double)input[(i+1)%n];output[6*i+2]=input[i]<-0.5;output[6*i+3]=input[i]==0.0;output[6*i+4]=input[i]<1.0e-12;output[6*i+5]=((double)input[i]+0.1)<(double)input[(i+1)%n];}`;
   const artifact=compile(source,{workgroupSize:[128]}),optimized=optimizeFloatComparisons(artifact);
   const edge=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f000000,0xbf000000,0x3effffff,0xbeffffff,0x3f000001,0xbf000001,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001];
   // Every ordered pair of edge values, plus reproducible random bit patterns.
   const bits=[];for(const a of edge)for(const b of edge)bits.push(a,b);let seed=0x12345678;for(let i=0;i<4096;i++){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;bits.push(seed>>>0);}
   const words=new Uint32Array(bits),values=new Float32Array(words.buffer),input=runtime.createBuffer(words),n=values.length,outputs=[];
   for(const a of [artifact,optimized.artifact]){const kernel=await runtime.kernel(a),output=runtime.createBuffer(n*6*4);runtime.batch().dispatch(kernel.bind({input,output},{n}),[Math.ceil(n/128)]).submit();outputs.push(await runtime.read(output,Uint32Array));runtime.destroyBuffer(output);}
   for(let i=0;i<outputs[0].length;i++)if(outputs[0][i]!==outputs[1][i])throw Error('Optimized mismatch at output '+i);
   // Independent JS binary64 reference for exact promotions and comparisons.
   for(let i=0;i<n;i++){const a=values[i],b=values[(i+1)%n],expected=[a<b,a===b,a<-.5,a===0,a<1e-12,(a+.1)<b];for(let j=0;j<6;j++)if(outputs[1][i*6+j]!==Number(expected[j]))throw Error('Reference mismatch at '+i+'/'+j+' bits '+words[i].toString(16));}
   if(!optimized.replacements)throw Error('No comparisons optimized');if(errors.length)throw Error(errors.join('\n'));runtime.destroyBuffer(input);
   return {pass:true,values:n,comparisons:n*6,replacements:optimized.replacements,exactGpuMatch:true,independentReference:true,includes:['NaNs','infinities','signed zero','subnormals','normal boundaries','nonrepresentable f64 constants','actual f64 arithmetic'],softwareAdapterRequested:false};
  }finally{runtime.dispose();}
 });writeFileSync('.local/wgsl-comparison-edges.json',JSON.stringify(result,null,2));console.log(result);
}finally{await browser.close();await new Promise(r=>server.close(r));}
