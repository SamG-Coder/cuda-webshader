import {createChronoRenderer} from './chrono-renderer.js';
import {createChronoLoop} from '../../tests/chrono-loop-plan.js';

// Sample host adapter. Every physics launch compiles the supplied editor source.
export async function chronoPreview({owner,container,runtime,source,compiler,onArtifact,onDispose,log,plan}){
 const load=async path=>{const response=await fetch(new URL('../../reports/'+path,import.meta.url));if(!response.ok)throw Error('Chrono input could not be loaded: '+path);return response;};
 const params=await(await load('chrono-params.json')).json(),positions=new Float32Array(await(await load('chrono-search-input.bin')).arrayBuffer()),properties=new Float32Array(await(await load('chrono-marker-rhopremu.bin')).arrayBuffer()),velocities=new Float32Array(await(await load('chrono-marker-velocities.bin')).arrayBuffer());
 const view=await createChronoRenderer({runtime,container,positions,properties});owner.renderer=view.renderer;
 onDispose(()=>{view.dispose();owner.renderer=null;});
 const original={pos:view.positions,rho:view.properties,vel:runtime.createBuffer(velocities)};owner.resources.push(original.vel);
 const kernelFactory=async(cuda,options)=>{if(cuda!==source)throw Error('Chrono launch source must match the editor');log('Compiling '+options.entry+' from editor CUDA…');const {artifact}=await compiler.compile(cuda,{...options,defines:{__CUDA_ARCH__:1,...options.defines}});onArtifact(artifact);return runtime.kernel(artifact);};
 const loop=await createChronoLoop(runtime,params,positions.length/4,{cudaSource:source,kernelFactory});onDispose(()=>loop.dispose());
 let steps=0,neighbors=0,failed=false;const dt=params['constant.paramsD.dT'];
 const advance=async()=>{const result=await loop(original);neighbors=result.neighbors;steps=result.steps;};
 log('All editor CUDA passes compiled. Advancing the first RK2 step…');
 const beforeRead=runtime.stats.readbackBytes;await advance();const controlReadbackBytes=runtime.stats.readbackBytes-beforeRead;const inspection=await runtime.read(original.pos,Float32Array,128*4);log('First RK2 step complete. Rendering shared GPU buffers…');view.render();
 // Readbacks yield to the browser between steps. Keep a single sequential pump
 // alive instead of waiting for a display frame to schedule every physics step.
 const pump=async()=>{while(owner.running&&!failed)await advance();};
 let frames=1,lastRender=-Infinity,lastMeasure=performance.now(),measuredSteps=steps,rate=0;
 view.renderer.setAnimationLoop(now=>{
  if(owner.running&&!owner.framePromise&&!failed){owner.framePromise=pump().catch(error=>{failed=true;owner.running=false;log(error.message,'error');}).finally(()=>{owner.framePromise=null;});}
  // Rendering every simulation step competes with compute on the same device.
  // Present at 30 fps while the solver runs at its own pace, retaining every step.
  if(now-lastRender<1000/30)return;
  lastRender=now;view.render();frames++;
  const elapsed=performance.now()-lastMeasure;
  if(elapsed>=1000){rate=(steps-measuredSteps)*1000/elapsed;measuredSteps=steps;lastMeasure=performance.now();}
  const info=document.getElementById('preview-info');if(info)info.textContent=`Chrono SPH · ${steps.toLocaleString()} steps · ${(steps*dt).toFixed(4)} simulated seconds · ${rate.toFixed(0)} steps/s · ${(rate*dt).toFixed(3)}× real time · ${neighbors.toLocaleString()} neighbours · colour = pressure`;
 });
 log('Original Chrono kernels running from editor source. Fixed step 0.0001 s; activity and neighbours rebuild each step. This experimental port runs slower than real time.','success');
 return {count:16731,inspection,controlReadbackBytes,get simulationSteps(){return steps;},get animationFrames(){return frames;},get steps(){return steps;},get simulatedSeconds(){return steps*dt;},buffers:original,settle:async()=>{await owner.framePromise;}};
}
