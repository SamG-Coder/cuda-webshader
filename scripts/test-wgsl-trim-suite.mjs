// Run the existing hardware suite with an opt-in post-emission WGSL pass.
import {createStaticServer} from './serve.mjs';
import {chromium} from 'playwright';
import {writeFileSync} from 'node:fs';
const server=createStaticServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage();page.on('console',m=>{if(m.type()==='log')console.log(m.text());});await page.goto(`http://127.0.0.1:${server.address().port}/`);
 await page.addScriptTag({type:'importmap',content:JSON.stringify({imports:{'three':'/node_modules/three/build/three.webgpu.js','three/webgpu':'/node_modules/three/build/three.webgpu.js','three/tsl':'/node_modules/three/build/three.tsl.js','three/addons/':'/node_modules/three/examples/jsm/'}})});
 const report=await page.evaluate(async numeric=>{
  const {GpuRuntime}=await import('/src/runtime/runtime.js'),{compile}=await import('/src/compiler/compiler.js'),{trimWgsl}=await import('/tests/experiments/wgsl-trim.js'),{runGpuSuite}=await import('/tests/gpu-suite.js'),{loadKernelSources}=await import('/src/kernels.js'),{runThreeInteropTest}=await import('/tests/three-interop.js');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(adapter.info.vendor!=='nvidia'||adapter.info.isFallbackAdapter)throw Error('Real NVIDIA required');
  const {optimizeFloatComparisons}=await import('/tests/experiments/wgsl-float-comparisons.js'),{optimizeFloatScaling}=await import('/tests/experiments/wgsl-float-scaling.js');
  const errors=[],runtime=await GpuRuntime.create({onError:e=>errors.push(e.message)}),originalKernel=runtime.kernel.bind(runtime),totals={artifacts:0,beforeBytes:0,afterBytes:0,projections:0,removedFunctions:0,comparisonReplacements:0,scalingReplacements:0};
  runtime.kernel=async(source,options={})=>{const result=trimWgsl(typeof source==='string'?compile(source,options):source);totals.artifacts++;for(const key of Object.keys(result.stats))totals[key]+=result.stats[key];let artifact=result.artifact;if(numeric){const a=optimizeFloatComparisons(artifact),b=optimizeFloatScaling(a.artifact);totals.comparisonReplacements+=a.replacements;totals.scalingReplacements+=b.replacements;artifact=b.artifact;}return originalKernel(artifact);};
  try{
   const report=await runGpuSuite(runtime,await loadKernelSources(),{onCase:(r,n)=>{if(n%20===0||!r.pass)console.log(n+': '+(r.pass?'PASS ':'FAIL ')+r.name);}});
   let interop;try{interop=await runThreeInteropTest(runtime);}catch(e){interop={name:'Three.js rendered-pixel interop',pass:false,error:String(e.stack||e)};}report.results.push(interop);report.total++;interop.pass?report.passed++:report.failed++;
   // Runtime scalar changes, including signed zero, must remain dynamic.
   const source='struct Settings { float x; float y; int mode; }; __constant__ Settings settings; __global__ void dynamic(float* output) { output[0]=settings.x; output[1]=settings.y; output[2]=settings.mode ? settings.x : settings.y; }';
   const kernel=await runtime.kernel(source,{workgroupSize:[1]}),output=runtime.createBuffer(12);
   let invocation;for(const [x,y,mode]of [[-0,2,0],[3,-0,1],[-7,11,0]]){const scalars={'constant.settings.x':x,'constant.settings.y':y,'constant.settings.mode':mode};if(invocation)invocation.setScalars(scalars);else invocation=kernel.bind({output},scalars);runtime.batch().dispatch(invocation,[1]).submit();const actual=await runtime.read(output,Uint32Array),expected=new Uint32Array(new Float32Array([x,y,mode?x:y]).buffer);if(actual.some((v,i)=>v!==expected[i]))throw Error('Trim froze dynamic uniforms or changed signed zero');}runtime.destroyBuffer(output);
   if(errors.length)throw Error(errors.join('\n'));
   return {...report,trim:totals,dynamicUniformsAndSignedZero:true,softwareAdapterRequested:false};
  }finally{runtime.dispose();}
 },process.argv.includes('--numeric'));
 writeFileSync(process.argv[2]??'.local/wgsl-trim-suite.json',JSON.stringify(report,null,2));console.log(JSON.stringify({total:report.total,passed:report.passed,failed:report.failed,trim:report.trim,dynamicUniformsAndSignedZero:report.dynamicUniformsAndSignedZero}));if(report.failed)process.exitCode=1;
}finally{await browser.close();await new Promise(r=>server.close(r));}
