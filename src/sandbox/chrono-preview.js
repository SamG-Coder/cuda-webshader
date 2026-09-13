import {createChronoRenderer} from './chrono-renderer.js';
import {createChronoRebuild} from '../../tests/chrono-rebuild.js';
import {createChronoStep} from '../../tests/chrono-step-plan.js';

// Sample host adapter. Every physics launch compiles the supplied editor source.
export async function chronoPreview({owner,container,runtime,source,compiler,onArtifact,onDispose,log,plan}){
 const load=async path=>{const response=await fetch(new URL('../../reports/'+path,import.meta.url));if(!response.ok)throw Error('Chrono input could not be loaded: '+path);return response;};
 const params=await(await load('chrono-params.json')).json(),positions=new Float32Array(await(await load('chrono-search-input.bin')).arrayBuffer()),properties=new Float32Array(await(await load('chrono-marker-rhopremu.bin')).arrayBuffer()),velocities=new Float32Array(await(await load('chrono-marker-velocities.bin')).arrayBuffer());
 const view=await createChronoRenderer({runtime,container,positions,properties});owner.renderer=view.renderer;
 onDispose(()=>{view.dispose();owner.renderer=null;});
 const original={pos:view.positions,rho:view.properties,vel:runtime.createBuffer(velocities)};owner.resources.push(original.vel);
 const kernelFactory=async(cuda,options)=>{if(cuda!==source)throw Error('Chrono launch source must match the editor');log('Compiling '+options.entry+' from editor CUDA…');const {artifact}=await compiler.compile(cuda,{...options,defines:{__CUDA_ARCH__:1,...options.defines}});onArtifact(artifact);return runtime.kernel(artifact);};
 const rebuild=await createChronoRebuild(runtime,params,positions.length/4,{cudaSource:source,kernelFactory});onDispose(()=>rebuild.dispose());
 const step=await createChronoStep(runtime,params,{cudaSource:source,kernelFactory});onDispose(()=>step.dispose());
 let steps=0,neighbors=0,failed=false;const dt=params['constant.paramsD.dT'];
 const advance=async()=>{const prepared=await rebuild(original,steps*.0001);try{neighbors=prepared.neighborEntries;await step(original,prepared);steps++;}finally{prepared.dispose();}};
 log('All editor CUDA passes compiled. Advancing the first RK2 step…');
 const beforeRead=runtime.stats.readbackBytes;await advance();const controlReadbackBytes=runtime.stats.readbackBytes-beforeRead;const inspection=await runtime.read(original.pos,Float32Array,128*4);log('First RK2 step complete. Rendering shared GPU buffers…');view.render();
 view.renderer.setAnimationLoop(()=>{if(owner.running&&!owner.framePromise&&!failed){owner.framePromise=advance().catch(error=>{failed=true;owner.running=false;log(error.message,'error');}).finally(()=>{owner.framePromise=null;});}view.render();const info=document.getElementById('preview-info');if(info)info.textContent=`Chrono SPH · ${steps.toLocaleString()} steps · ${(steps*dt).toFixed(4)} simulated seconds · ${neighbors.toLocaleString()} neighbours · colour = pressure`;});
 log('Original Chrono kernels running from editor source. Fixed step 0.0001 s; activity and neighbours rebuild each step. This experimental port runs slower than real time.','success');
 return {count:16731,inspection,controlReadbackBytes,get simulationSteps(){return steps;},get animationFrames(){return steps;},get steps(){return steps;},get simulatedSeconds(){return steps*dt;},buffers:original,settle:async()=>{await owner.framePromise;}};
}
