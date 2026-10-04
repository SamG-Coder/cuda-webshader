// Real native permission + NVRTC/CUDA execution, followed by real WebGPU.
// Permission decisions use CDP; this does not claim a human clicked the prompt.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile,access} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createStaticServer} from './serve.mjs';
const executablePath=process.env.RTXCUDA_CHROME||'D:/ChromiumRTXCuda/src/out/RTXCuda/chrome.exe';
await access(executablePath);
const server=createStaticServer(process.env.NATIVE_TEST_ROOT);await new Promise(r=>server.listen(0,'127.0.0.1',r));
const origin=`http://127.0.0.1:${server.address().port}`,report={executablePath,permissionDecisions:'DevTools-controlled, native browser enforcement',checks:[]};
await mkdir('test-results',{recursive:true});let browser;
const check=name=>{report.checks.push(name);console.log('PASS:',name);};
try{
 browser=await chromium.launch({executablePath,headless:true,chromiumSandbox:true,args:['--no-first-run','--no-default-browser-check']});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.route('**/native-integration',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><title>WebCuda native backend integration</title><button id="enable">Enable native CUDA</button><pre id="result"></pre>'}));
 await page.goto(origin+'/native-integration');const cdp=await page.context().newCDPSession(page);
 const {targetInfo}=await cdp.send('Target.getTargetInfo');
 const permission=setting=>cdp.send('Browser.setPermission',{permission:{name:'native-gpu'},setting,origin,browserContextId:targetInfo.browserContextId});
 await page.evaluate(async()=>{
  window.api=await import('/src/runtime/runtime.js');window.compiler=await import('/src/compiler/compiler.js');
  window.runCompute=async(runtime,artifact=true)=>{
   const n=262147,x=new Float32Array(n),y=new Float32Array(n);for(let i=0;i<n;i++){x[i]=(i%127)*.25;y[i]=(i%31)-10;}
   const a=runtime.createBuffer(x),b=runtime.createBuffer(y);x.fill(-999);y.fill(-999);
   const source='__global__ void saxpy(float factor,const float* x,float* y,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=factor*x[i]+y[i];}';
   const compiled=artifact?JSON.parse(JSON.stringify(compiler.serializableArtifact(compiler.compile(source,{entry:'saxpy',workgroupSize:[128,1,1],includeNativeSource:true})))):source;
   const kernel=await runtime.kernel(compiled,{entry:'saxpy',workgroupSize:[128,1,1]}),inv=kernel.bind({y:b,x:a},{n,factor:2}),batch=runtime.batch().dispatch(inv,[Math.ceil(n/128)]);
   inv.setScalars({factor:3});batch.dispatch(inv,[Math.ceil(n/128)]).submit();const output=await runtime.read(b);let bad=0;
   for(let i=0;i<n;i++)if(output[i]!==5*(i%127)*.25+(i%31)-10)bad++;
   const update=new Float32Array([7,8,9,10]);runtime.write(b,update,524284);update.fill(0);const partial=Array.from(await runtime.read(b,Float32Array,16,524284));
   const result={backend:runtime.backend,describe:runtime.describe(),values:n,mismatches:bad,partial,stats:{...runtime.stats}};
   runtime.destroyBuffer(a);runtime.destroyBuffer(b);await runtime.idle();await runtime.dispose();return result;
  };
  document.querySelector('#enable').onclick=()=>{window.clicked=(async()=>{const state=await api.GpuRuntime.requestPermission();if(state!=='granted')throw Error(state);return runCompute(await api.GpuRuntime.create());})();};
 });
 assert.equal(await page.evaluate(()=>api.SupportsNativeCuda()),true);check('fork exposes real native CUDA capability');
 await permission('prompt');
 // Expire Playwright's transient activation, then use CDP without a gesture.
 let activated=true;for(let i=0;i<65&&activated;i++){activated=(await cdp.send('Runtime.evaluate',{expression:'navigator.userActivation.isActive',returnByValue:true,userGesture:false})).result.value;if(activated)await new Promise(r=>setTimeout(r,100));}
 assert.equal(activated,false);
 const noGesture=await cdp.send('Runtime.evaluate',{expression:`api.GpuRuntime.create().then(async r=>{const result={backend:r.backend,reason:r.describe().native.reason};await r.dispose();return result;})`,awaitPromise:true,returnByValue:true,userGesture:false});
 assert.deepEqual(noGesture.result.value,{backend:'webgpu',reason:'permission-required'});check('no-gesture startup falls back without opening a permission prompt');
 const noGestureRequest=await cdp.send('Runtime.evaluate',{expression:`api.requestPermission().then(value=>({value}),e=>({error:e.name}))`,awaitPromise:true,returnByValue:true,userGesture:false});
 assert.equal(noGestureRequest.result.value.error,'NotAllowedError');check('permission helper respects the native user-gesture gate');
 await permission('denied');
 assert.equal(await page.evaluate(async()=>{try{await api.GpuRuntime.create({backend:'native'});}catch(e){return e.name;}}),'NotAllowedError');
 report.denied=await page.evaluate(async()=>runCompute(await api.GpuRuntime.create()));assert.equal(report.denied.backend,'webgpu');assert.equal(report.denied.mismatches,0);assert.deepEqual(report.denied.partial,[7,8,9,10]);check('denial falls back to real WebGPU; strict native rejects');
 await permission('granted');assert.equal(await page.evaluate(()=>navigator.cuda.queryPermission()),'granted');await page.locator('#enable').click();report.native=await page.evaluate(()=>clicked);
 assert.equal(report.native.backend,'native-cuda');assert.equal(report.native.mismatches,0);assert.deepEqual(report.native.partial,[7,8,9,10]);check('public GpuRuntime click flow executes dual artifact on CUDA with 262147 checked values');
 report.raw=await page.evaluate(async()=>runCompute(await api.GpuRuntime.create(),false));assert.equal(report.raw.backend,'native-cuda');assert.equal(report.raw.mismatches,0);check('unmodified raw CUDA API, chunked snapshots, named ABI and two dispatch scalar versions');
 report.explicit=await page.evaluate(async()=>runCompute(await api.GpuRuntime.create({backend:'webgpu'})));assert.equal(report.explicit.backend,'webgpu');assert.equal(report.explicit.mismatches,0);check('explicit WebGPU works even when native is supported and permitted');
 await page.evaluate(async()=>{window.live=await api.GpuRuntime.create();window.buffer=live.createBuffer(new Float32Array([37]));await live.idle();});await permission('denied');
 const revoked=await page.evaluate(async()=>{try{await live.read(buffer);return 'unexpected success';}catch(e){return e.message;}finally{await live.dispose();}});assert.match(revoked,/permission|revoked|closed/i);check('native permission revocation rejects live resources without switching backends');
 assert.deepEqual(errors,[]);report.errors=errors;await browser.close();browser=null;
 browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});const edge=await browser.newPage();await edge.route('**/native-integration',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><title>Ordinary browser fallback</title>'}));await edge.goto(origin+'/native-integration');
 report.edge=await edge.evaluate(async()=>{const {GpuRuntime,SupportsNativeCuda}=await import('/src/runtime/runtime.js');const available=await SupportsNativeCuda(),r=await GpuRuntime.create();const b=r.createBuffer(new Float32Array([1,2,3,4])),k=await r.kernel('__global__ void k(float* a){unsigned i=threadIdx.x;if(i<4)a[i]*=3;}');r.batch().dispatch(k.bind({a:b}),[1]).submit();const out=Array.from(await r.read(b)),backend=r.backend;r.dispose();return {available,backend,out};});
 assert.deepEqual(report.edge,{available:false,backend:'webgpu',out:[3,6,9,12]});check('stock Edge requires no native permission and retains WebGPU execution');
 await writeFile('test-results/native-browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
