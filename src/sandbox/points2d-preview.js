// SPDX-License-Identifier: MIT
// Draw normalized float2 compute output directly as one-pixel GPU points.
import {executePipeline} from './pipeline.js';
export async function points2DPreview({owner,container,plan,runtime,...options}){
 const p=plan.preview,canvas=document.createElement('canvas');canvas.style.cssText='width:100%;height:100%;touch-action:none;cursor:crosshair';container.replaceChildren(canvas);
 const before=runtime.stats.readbackBytes,result=await executePipeline({...options,plan,runtime,resources:owner.resources,textureResources:owner.textureResources});
 const device=runtime.device,context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
 const module=device.createShaderModule({code:'@group(0) @binding(0) var<storage,read> positions:array<vec2<f32>>; @vertex fn vertex(@builtin(vertex_index) id:u32)->@builtin(position) vec4<f32>{let p=positions[id];return vec4<f32>(p.x*2-1,1-p.y*2,0,1);} @fragment fn fragment()->@location(0) vec4<f32>{return vec4<f32>(0,1,0,0.5);}'});
 const pipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:[{format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'}}}]},primitive:{topology:'point-list'}});
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:result.buffers[p.positions].gpuBuffer}}]});
 let stopped=false,raf=0,dirty=true,pointer=null,pending=null,interactionFrames=0,draws=0;
 const resize=()=>{canvas.width=Math.max(1,container.clientWidth);canvas.height=Math.max(1,container.clientHeight);dirty=true;};owner.observer=new ResizeObserver(resize);owner.observer.observe(container);resize();
 const draw=()=>{const e=device.createCommandEncoder(),pass=e.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(p.count);pass.end();device.queue.submit([e.finish()]);draws++;dirty=false;};
 const stir=p.stir,force=stir?plan.steps[stir.step].scalars:null,original=force?{...force}:null;
 const coordinate=e=>{const b=canvas.getBoundingClientRect();return {x:(e.clientX-b.left)/b.width,y:(e.clientY-b.top)/b.height};};
 canvas.onpointerdown=e=>{if(!stir)return;pointer=coordinate(e);canvas.setPointerCapture(e.pointerId);};
 canvas.onpointermove=e=>{if(!pointer||!stir)return;const next=coordinate(e),r=stir.radius;
  pending={x:Math.max(0,Math.min(stir.width-2*r-1,Math.floor(pointer.x*stir.width)-r)),y:Math.max(0,Math.min(stir.height-2*r-1,Math.floor(pointer.y*stir.height)-r)),fx:(pending?.fx||0)+(next.x-pointer.x)*stir.scale,fy:(pending?.fy||0)+(next.y-pointer.y)*stir.scale};pointer=next;};
 canvas.onpointerup=canvas.onpointercancel=()=>{pointer=null;};
 const loop=()=>{if(stopped)return;raf=requestAnimationFrame(loop);if(owner.framePromise)return;if(!owner.running){if(dirty)draw();return;}
  owner.framePromise=(async()=>{const change=pending;pending=null;if(change)Object.assign(force,{[stir.x]:change.x,[stir.y]:change.y,[stir.fx]:change.fx,[stir.fy]:change.fy});try{await result.stepFrame();if(change)interactionFrames++;owner.animationFrames=(owner.animationFrames||0)+1;}finally{if(change)Object.assign(force,original);}if(!stopped)draw();})().catch(e=>{owner.running=false;options.log?.(e.message,'error');}).finally(()=>{owner.framePromise=null;});};
 owner.renderer={setAnimationLoop(callback){if(callback===null){stopped=true;cancelAnimationFrame(raf);}},dispose(){context.unconfigure();canvas.onpointerdown=canvas.onpointermove=canvas.onpointerup=canvas.onpointercancel=null;}};
 draw();raf=requestAnimationFrame(loop);
 return {...result,inspection:new Float32Array(),controlReadbackBytes:runtime.stats.readbackBytes-before,get simulationSteps(){return result.steps;},get animationFrames(){return owner.animationFrames||0;},get interactionFrames(){return interactionFrames;},get draws(){return draws;},settle:async()=>{await owner.framePromise;}};
}
