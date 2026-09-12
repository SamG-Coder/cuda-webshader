import {chromium} from 'playwright';
import {createStaticServer} from './serve.mjs';
import {writeFile} from 'node:fs/promises';
const server=createStaticServer(process.cwd());await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
try{
 const page=await browser.newPage({viewport:{width:900,height:750}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:'+server.address().port+'/sandbox.html?example=blank');
 await page.waitForFunction(()=>window.sandbox?.completedRuns||window.sandbox?.lastError,null,{timeout:60000});
 const result=await page.evaluate(async()=>{
  const {createSmokeRenderer}=await import('/src/sandbox/smoke-renderer.js'),{PerspectiveCamera}=await import('three/webgpu'),runtime=window.sandbox.runtime;
  if(runtime.describe().vendor!=='nvidia')throw Error('Real NVIDIA required');
  const count=8192,positions=new Float32Array(count*4),velocities=new Float32Array(count*4),indices=Uint32Array.from({length:count},(_,i)=>i);
  let seed=123;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  for(let i=0;i<count;i++){const theta=random()*Math.PI*2,z=random()*2-1,r=Math.cbrt(random()),xy=Math.sqrt(1-z*z);positions.set([Math.cos(theta)*xy*r, z*r,Math.sin(theta)*xy*r,0],i*4);velocities[i*4+3]=100;}
  const p=runtime.createBuffer(positions),v=runtime.createBuffer(velocities),ix=runtime.createBuffer(indices),keys=runtime.createBuffer(count*4);
  const canvas=document.createElement('canvas');canvas.id='smoke-render-test';document.body.replaceChildren(canvas);
  const renderer=await createSmokeRenderer({device:runtime.device,canvas,positions:p,velocities:v,indices:ix,count,radius:.075});renderer.resize(768,640);
  const camera=new PerspectiveCamera(40,768/640,.1,100);camera.coordinateSystem=2001;camera.position.set(3,2,4);camera.lookAt(0,0,0);camera.updateProjectionMatrix();camera.updateMatrixWorld();
  const source=await(await fetch('/tests/smoke-integration-kernel.cuh')).text(),depth=await runtime.kernel(source,{entry:'calcDepth_functor',workgroupSize:[128,1,1]});
  const sort=async()=>{const direction=renderer.orientation(camera).direction;runtime.device.queue.writeBuffer(ix.gpuBuffer,0,indices);runtime.batch().dispatch(depth.bind({tuple0:p,tuple1:keys},{count,'sortVector.x':direction.x,'sortVector.y':direction.y,'sortVector.z':direction.z}),[64,1,1]).submit();await runtime.sortPairs(keys,ix,{count,keyType:'f32'});};
  const read=async()=>{const width=768,height=640,bytesPerRow=width*8,staging=runtime.device.createBuffer({size:bytesPerRow*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=runtime.device.createCommandEncoder();e.copyTextureToBuffer({texture:renderer.image},{buffer:staging,bytesPerRow},{width,height});runtime.device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);const data=new Uint16Array(staging.getMappedRange()).slice();staging.unmap();staging.destroy();return data;};
  await sort();renderer.render(camera,{shadows:false});const plain=await read();renderer.render(camera);const shaded=await read();
  let changed=0,covered=0,invalid=0;for(let i=0;i<plain.length;i+=4){if(shaded[i+3])covered++;if(plain[i]!==shaded[i]||plain[i+1]!==shaded[i+1]||plain[i+2]!==shaded[i+2])changed++;for(let c=0;c<4;c++)if((shaded[i+c]&0x7c00)===0x7c00)invalid++;}
  if(changed<1000||covered<1000||invalid)throw Error('Invalid smoke pixels '+JSON.stringify({changed,covered,invalid}));
  const initialInverted=renderer.orientation(camera).inverted;camera.position.set(-3,-2,-4);camera.lookAt(0,0,0);camera.updateMatrixWorld();await sort();renderer.render(camera);await runtime.idle();
  if(renderer.orientation(camera).inverted===initialInverted)throw Error('Opposite view did not exercise reverse blending');
  camera.position.set(3,2,4);camera.lookAt(0,0,0);camera.updateMatrixWorld();await sort();renderer.render(camera);await runtime.idle();
  window.smokeRendererTest={renderer,runtime,resources:[p,v,ix,keys]};
  return {passed:true,count,slices:32,changedShadowPixels:changed,coveredPixels:covered,invalidPixels:invalid,bothBlendDirections:true,device:runtime.describe(),softwareAdapterRequested:false};
 });
 await page.locator('#smoke-render-test').screenshot({path:'reports/smoke-renderer-preview.png'});
 await page.evaluate(async()=>{const {renderer,runtime,resources}=window.smokeRendererTest;await runtime.idle();renderer.dispose();resources.forEach(r=>runtime.destroyBuffer(r));});
 if(errors.length)throw Error(errors.join('\n'));
 await writeFile('reports/smoke-renderer-check.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
