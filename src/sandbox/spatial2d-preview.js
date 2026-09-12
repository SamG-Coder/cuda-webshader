// SPDX-License-Identifier: MIT
// Generic preview of scalar point coordinates and rectangular storage records.
import {executePipeline} from './pipeline.js';
export async function spatial2DPreview({owner,container,plan,runtime,...options}){
 const before=runtime.stats.readbackBytes,result=await executePipeline({...options,plan,runtime,resources:owner.resources,textureResources:owner.textureResources}),p=plan.preview;
 const canvas=document.createElement('canvas');canvas.style.cssText='width:100%;height:100%;touch-action:none;cursor:grab';container.replaceChildren(canvas);
 const device=runtime.device,context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
 const module=device.createShaderModule({code:`
 @group(0) @binding(0) var<storage,read> records:array<u32>;
 @group(0) @binding(1) var<storage,read> xs:array<f32>;
 @group(0) @binding(2) var<storage,read> ys:array<f32>;
 struct View { bounds:vec4<f32>, pixels:vec4<f32> }
 @group(0) @binding(3) var<uniform> view:View;
 struct Vertex { @builtin(position) position:vec4<f32>, @location(0) color:vec3<f32> }
 fn project(p:vec2<f32>)->vec4<f32>{let uv=(p-view.bounds.xy)/view.bounds.zw;return vec4<f32>(uv.x*2-1,1-uv.y*2,0,1);}
 @vertex fn box(@builtin(vertex_index) id:u32,@builtin(instance_index) node:u32)->Vertex{
  var out:Vertex;out.position=vec4<f32>(2,2,0,1);out.color=vec3<f32>(0);
  let base=node*${p.recordWords}u;if(records[base+${p.endWord}u]<=records[base+${p.beginWord}u]){return out;}
  let lo=vec2<f32>(bitcast<f32>(records[base+${p.boundsWords[0]}u]),bitcast<f32>(records[base+${p.boundsWords[1]}u]));
  let hi=vec2<f32>(bitcast<f32>(records[base+${p.boundsWords[2]}u]),bitcast<f32>(records[base+${p.boundsWords[3]}u]));
  let corners=array<vec2<f32>,8>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(1,0),vec2<f32>(1,1),vec2<f32>(1,1),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(0,0));
  out.position=project(mix(lo,hi,corners[id]));let level=-log2(max(hi.x-lo.x,0.000001));out.color=vec3<f32>(0.35)+0.25*cos(vec3<f32>(level,level+2.1,level+4.2));return out;
 }
 @vertex fn point(@builtin(vertex_index) id:u32,@builtin(instance_index) index:u32)->Vertex{
  let corners=array<vec2<f32>,6>(vec2<f32>(-1,-1),vec2<f32>(1,-1),vec2<f32>(1,1),vec2<f32>(-1,-1),vec2<f32>(1,1),vec2<f32>(-1,1));var out:Vertex;out.position=project(vec2<f32>(xs[index],ys[index]));out.position+=vec4<f32>(corners[id]*3/view.pixels.xy,0,0);out.color=vec3<f32>(0.65,1,0.85);return out;
 }
 @fragment fn fragment(in:Vertex)->@location(0) vec4<f32>{return vec4<f32>(in.color,1);}`});
 const layout=device.createBindGroupLayout({entries:[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.VERTEX,buffer:{type:'read-only-storage'}})).concat([{binding:3,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform'}}])}),pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
 const pipelines=await Promise.all(['box','point'].map(entry=>device.createRenderPipelineAsync({layout:pipelineLayout,vertex:{module,entryPoint:entry},fragment:{module,entryPoint:'fragment',targets:[{format}]},primitive:{topology:entry==='box'?'line-list':'triangle-list'}})));
 const uniform=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});owner.disposers.push(()=>uniform.destroy());
 const group=device.createBindGroup({layout,entries:[p.boxes,p.x,p.y].map((name,binding)=>({binding,resource:{buffer:result.buffers[name].gpuBuffer}})).concat([{binding:3,resource:{buffer:uniform}}])});
 let bounds=[...p.bounds],stopped=false,drag=null,draws=0;
 const draw=()=>{if(stopped)return;device.queue.writeBuffer(uniform,0,new Float32Array([bounds[0],bounds[1],bounds[2]-bounds[0],bounds[3]-bounds[1],canvas.width,canvas.height,0,0]));const e=device.createCommandEncoder(),pass=e.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:.015,g:.035,b:.05,a:1}}]});pass.setBindGroup(0,group);pass.setPipeline(pipelines[0]);pass.draw(8,p.boxCount);pass.setPipeline(pipelines[1]);pass.draw(6,p.count);pass.end();device.queue.submit([e.finish()]);draws++;};
 const resize=()=>{canvas.width=Math.max(1,container.clientWidth);canvas.height=Math.max(1,container.clientHeight);draw();};owner.observer=new ResizeObserver(resize);owner.observer.observe(container);
 canvas.onwheel=e=>{e.preventDefault();const f=Math.exp(Math.max(-1,Math.min(1,e.deltaY*.001))),cx=(bounds[0]+bounds[2])/2,cy=(bounds[1]+bounds[3])/2,w=Math.min(100,Math.max(.001,(bounds[2]-bounds[0])*f)),h=Math.min(100,Math.max(.001,(bounds[3]-bounds[1])*f));bounds=[cx-w/2,cy-h/2,cx+w/2,cy+h/2];draw();};
 canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag)return;const dx=(e.clientX-drag[0])/canvas.clientWidth*(bounds[2]-bounds[0]),dy=(e.clientY-drag[1])/canvas.clientHeight*(bounds[3]-bounds[1]);bounds=[bounds[0]-dx,bounds[1]-dy,bounds[2]-dx,bounds[3]-dy];drag=[e.clientX,e.clientY];draw();};canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};canvas.ondblclick=()=>{bounds=[...p.bounds];draw();};
 owner.renderer={setAnimationLoop(callback){if(callback===null)stopped=true;},dispose(){context.unconfigure();canvas.onwheel=canvas.onpointerdown=canvas.onpointermove=canvas.onpointerup=canvas.onpointercancel=canvas.ondblclick=null;}};
 resize();return {...result,inspection:new Float32Array(),controlReadbackBytes:runtime.stats.readbackBytes-before,geometryReadbackBytes:0,get draws(){return draws;},settle:()=>runtime.idle()};
}
