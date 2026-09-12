// SPDX-License-Identifier: MIT
// Render float2 device allocations directly. CUDA computes all curve vertices.
import {executePipeline} from './pipeline.js';
export async function allocatedCurvesPreview({owner,container,plan,runtime,...options}){
 const before=runtime.stats.readbackBytes,result=await executePipeline({...options,plan,runtime,resources:owner.resources,textureResources:owner.textureResources}),p=plan.preview;
 const poolIndex=result.pools.findIndex(t=>t.name===p.pool),pool=result.pools[poolIndex];if(!pool)throw Error('Allocated curve pool is missing.');
 const layout=JSON.parse(pool.recordLayout);if(layout.element!=='vec2<f32>'||layout.stride!==8||!Number.isInteger(layout.maxElements)||layout.maxElements<2)throw Error('Allocated curves require a float2 device pool.');
 const canvas=document.createElement('canvas');canvas.style.cssText='width:100%;height:100%;touch-action:none;cursor:grab';container.replaceChildren(canvas);
 const device=runtime.device,context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
 const module=device.createShaderModule({code:`
 @group(0) @binding(0) var<storage,read> records:array<u32>;
 @group(0) @binding(1) var<storage,read> pool:array<u32>;
 @group(0) @binding(2) var<uniform> view:vec4<f32>;
 struct Vertex { @builtin(position) position:vec4<f32>, @location(0) color:vec3<f32> }
 @vertex fn vertex(@builtin(vertex_index) id:u32,@builtin(instance_index) curve:u32)->Vertex {
   var out:Vertex;out.position=vec4<f32>(2,2,0,1);out.color=vec3<f32>(0);
   let handle=records[curve*${p.recordWords}u+${p.pointerWord}u];
   if(handle==0u||handle>${layout.maxAllocations}u){return out;}
   let count=pool[handle-1u];let segment=id/2u;let point=segment+id%2u;
   if(segment+1u>=count||count>${layout.maxElements}u){return out;}
   let base=${layout.offset/4}u+((handle-1u)*${layout.maxElements}u+point)*2u;
   let xy=vec2<f32>(bitcast<f32>(pool[base]),bitcast<f32>(pool[base+1u]));
   let uv=(xy-view.xy)/view.zw;out.position=vec4<f32>(uv.x*2-1,1-uv.y*2,0,1);
   let hue=f32(curve)*2.39996323;out.color=vec3<f32>(0.55)+0.4*cos(vec3<f32>(hue,hue+2.094,hue+4.188));return out;
 }
 @fragment fn fragment(in:Vertex)->@location(0) vec4<f32>{return vec4<f32>(in.color,0.85);}`});
 const pipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:[{format}]},primitive:{topology:'line-list'}});
 const uniform=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});owner.disposers.push(()=>uniform.destroy());
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:result.buffers[p.records].gpuBuffer}},{binding:1,resource:{buffer:result.objectArena.buffers[poolIndex].gpuBuffer}},{binding:2,resource:{buffer:uniform}}]});
 let bounds=[...p.bounds],stopped=false,drag=null,draws=0;
 const draw=()=>{if(stopped)return;device.queue.writeBuffer(uniform,0,new Float32Array([bounds[0],bounds[1],bounds[2]-bounds[0],bounds[3]-bounds[1]]));const e=device.createCommandEncoder(),pass=e.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:.015,g:.035,b:.05,a:1}}]});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw((layout.maxElements-1)*2,p.count);pass.end();device.queue.submit([e.finish()]);draws++;};
 const resize=()=>{canvas.width=Math.max(1,container.clientWidth);canvas.height=Math.max(1,container.clientHeight);draw();};owner.observer=new ResizeObserver(resize);owner.observer.observe(container);
 canvas.onwheel=e=>{e.preventDefault();const f=Math.exp(Math.max(-1,Math.min(1,e.deltaY*.001))),cx=(bounds[0]+bounds[2])/2,cy=(bounds[1]+bounds[3])/2,w=Math.min(100,Math.max(.001,(bounds[2]-bounds[0])*f)),h=Math.min(100,Math.max(.001,(bounds[3]-bounds[1])*f));bounds=[cx-w/2,cy-h/2,cx+w/2,cy+h/2];draw();};
 canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag)return;const dx=(e.clientX-drag[0])/canvas.clientWidth*(bounds[2]-bounds[0]),dy=(e.clientY-drag[1])/canvas.clientHeight*(bounds[3]-bounds[1]);bounds=[bounds[0]-dx,bounds[1]-dy,bounds[2]-dx,bounds[3]-dy];drag=[e.clientX,e.clientY];draw();};canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};canvas.ondblclick=()=>{bounds=[...p.bounds];draw();};
 owner.renderer={setAnimationLoop(callback){if(callback===null)stopped=true;},dispose(){context.unconfigure();canvas.onwheel=canvas.onpointerdown=canvas.onpointermove=canvas.onpointerup=canvas.onpointercancel=canvas.ondblclick=null;}};
 resize();return {...result,inspection:new Float32Array(),controlReadbackBytes:runtime.stats.readbackBytes-before,geometryReadbackBytes:0,get draws(){return draws;},settle:()=>runtime.idle()};
}
