import {PerspectiveCamera,Vector3,WebGPUCoordinateSystem} from 'three/webgpu';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createSmokeRenderer} from './smoke-renderer.js';
import {executePipeline} from './pipeline.js';
export async function smokePreview({owner,container,plan,runtime,...options}){
 const p=plan.preview,canvas=document.createElement('canvas');canvas.style.width='100%';canvas.style.height='100%';container.replaceChildren(canvas);
 const camera=new PerspectiveCamera(40,1,.1,100);camera.coordinateSystem=WebGPUCoordinateSystem;camera.position.set(3,2,4);camera.lookAt(0,0,0);
 const controls=owner.controls=new OrbitControls(camera,canvas);controls.enableDamping=true;
 const depthStep=plan.steps[p.depthStep],indexStep=plan.steps[p.indexStep];
 const direction=()=>{camera.updateMatrixWorld();const view=camera.getWorldDirection(new Vector3()),light=new Vector3(5,5,-5).normalize();return light.add(view.multiplyScalar(view.dot(light)<=0?-1:1)).normalize();};
 const updateDirection=()=>{const d=direction();Object.assign(depthStep.scalars,{'sortVector.x':d.x,'sortVector.y':d.y,'sortVector.z':d.z});};updateDirection();
 const before=runtime.stats.readbackBytes;
 const result=await executePipeline({...options,plan,runtime,resources:owner.resources,textureResources:owner.textureResources});
 const renderer=await createSmokeRenderer({device:runtime.device,canvas,positions:result.buffers[p.positions],velocities:result.buffers[p.velocities],indices:result.buffers[p.indices],count:p.count,radius:p.radius,slices:32});
 let stopped=false,raf=0;
 owner.renderer={setAnimationLoop(callback){if(callback===null){stopped=true;cancelAnimationFrame(raf);}},dispose(){renderer.dispose();}};
 const depthKernel=await runtime.kernel(options.source,{entry:depthStep.entry,workgroupSize:depthStep.block}),indexKernel=await runtime.kernel(options.source,{entry:indexStep.entry,workgroupSize:indexStep.block});
 const bind=(step,kernel)=>kernel.bind(Object.fromEntries(kernel.artifact.metadata.bindings.map(b=>[b.name,result.buffers[step.bindings?.[b.name]??b.name]])),step.scalars);
 const sort=async()=>{runtime.batch().dispatch(bind(depthStep,depthKernel),depthStep.groups).dispatch(bind(indexStep,indexKernel),indexStep.groups).submit();await runtime.sortPairs(result.buffers[p.depth],result.buffers[p.indices],{count:p.count,keyType:'f32'});};
 let lastView='',frame=1;
 const resize=()=>{const w=Math.max(1,Math.floor(container.clientWidth)),h=Math.max(1,Math.floor(container.clientHeight));camera.aspect=w/h;camera.updateProjectionMatrix();renderer.resize(w,h);lastView='';};owner.observer=new ResizeObserver(resize);owner.observer.observe(container);resize();
 renderer.render(camera);
 const loop=()=>{
  if(stopped)return;raf=requestAnimationFrame(loop);if(owner.framePromise)return;controls.update();camera.updateMatrixWorld();const view=camera.matrixWorld.elements.join(',');
  if(!owner.running&&view===lastView)return;
  updateDirection();lastView=view;
  owner.framePromise=(async()=>{if(owner.running&&plan.animate){await result.stepFrame(frame*.5);frame++;owner.animationFrames=(owner.animationFrames||0)+1;}else await sort();if(!stopped)renderer.render(camera);})().catch(e=>{owner.running=false;options.log?.(e.message,'error');}).finally(()=>{owner.framePromise=null;});
 };raf=requestAnimationFrame(loop);
 return {...result,inspection:new Float32Array(),controlReadbackBytes:runtime.stats.readbackBytes-before,get simulationSteps(){return result.steps;},get animationFrames(){return owner.animationFrames||0;},settle:async()=>{await owner.framePromise;},smokeRenderer:renderer};
}
