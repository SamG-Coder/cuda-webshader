// SPDX-License-Identifier: BSD-3-Clause
// WebGPU adaptation of NVIDIA smokeParticles/SmokeShaders.cpp and
// SmokeRenderer.cpp (Copyright (c) 2022 NVIDIA CORPORATION).
// See licenses/nvidia-cuda-samples-BSD-3-Clause.txt and THIRD_PARTY_NOTICES.md.
import {PerspectiveCamera,Matrix4,Vector3,WebGPUCoordinateSystem} from 'three/webgpu';

const shader=String.raw`
struct Uniforms {view:mat4x4<f32>,projection:mat4x4<f32>,inverseView:mat4x4<f32>,lightVP:mat4x4<f32>,settings:vec4<f32>,mode:vec4<f32>}
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> velocities:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> indices:array<u32>;
@group(0) @binding(3) var<uniform> u:Uniforms;
@group(1) @binding(0) var attenuation:texture_2d<f32>;
@group(1) @binding(1) var linearSampler:sampler;
struct Vertex { @builtin(position) clip:vec4<f32>, @location(0) uv:vec2<f32>, @location(1) eye:vec3<f32>, @location(2) fade:f32 }
@vertex fn vertex(@builtin(vertex_index) v:u32,@builtin(instance_index) instance:u32)->Vertex {
 let id=indices[instance];let p=positions[id];let velocity=velocities[id];
 let eye=(u.view*vec4<f32>(p.xyz,1)).xyz;
 var previous=(u.view*vec4<f32>(p.xyz-velocity.xyz*u.settings.y,1)).xyz;
 let motion=eye-previous;let distance=length(motion);
 var x=vec3<f32>(u.settings.x,0,0);var y=vec3<f32>(0,-u.settings.x,0);
 if(distance>0.01){
  let direction=motion/distance;let view=normalize(-eye);let facing=dot(view,direction);
  if(abs(facing)<0.95){x=direction*u.settings.x;y=normalize(cross(direction,view))*u.settings.x;}else{previous=eye;}
 }else{previous=eye;}
 let corners=array<vec2<f32>,6>(vec2<f32>(-1,-1),vec2<f32>(1,-1),vec2<f32>(-1,1),vec2<f32>(-1,1),vec2<f32>(1,-1),vec2<f32>(1,1));
 let corner=corners[v];let center=select(previous,eye,corner.x>0);
 let point=center+corner.x*x+corner.y*y;
 var result:Vertex;result.clip=u.projection*vec4<f32>(point,1);result.eye=point;result.uv=corner;
 let phase=select(1.0,p.w/max(velocity.w,0.000001),velocity.w>0);
 result.fade=clamp(1-phase,0,1);return result;
}
@fragment fn shadowed(input:Vertex)->@location(0) vec4<f32>{
 let r2=dot(input.uv,input.uv);if(r2>1){discard;}
 let normal=vec3<f32>(input.uv.x,-input.uv.y,sqrt(1-r2));
 let world=u.inverseView*vec4<f32>(input.eye+normal*u.settings.x,1);
 let light=u.lightVP*world;let uv=light.xy/light.w*vec2<f32>(0.5,-0.5)+vec2<f32>(0.5);
 let shadow=vec3<f32>(1)-textureSampleLevel(attenuation,linearSampler,uv,0).xyz;
 let alpha=(1-r2)*input.fade*u.settings.z;
 return vec4<f32>(shadow*alpha,alpha);
}
@fragment fn lightParticle(input:Vertex)->@location(0) vec4<f32>{
 let r2=dot(input.uv,input.uv);if(r2>1){discard;}
 let alpha=(1-r2)*input.fade;
 return vec4<f32>(vec3<f32>(0.1,0.2,0.3)*u.settings.w*alpha,alpha);
}
`;
const composite=String.raw`
@group(0) @binding(0) var image:texture_2d<f32>;
@group(0) @binding(1) var linearSampler:sampler;
struct V{@builtin(position) pos:vec4<f32>,@location(0) uv:vec2<f32>}
@vertex fn vertex(@builtin(vertex_index) i:u32)->V{
 let p=array<vec2<f32>,3>(vec2<f32>(-1,-1),vec2<f32>(3,-1),vec2<f32>(-1,3));var v:V;v.pos=vec4<f32>(p[i],0,1);v.uv=p[i]*vec2<f32>(0.5,-0.5)+vec2<f32>(0.5);return v;
}
@fragment fn fragment(v:V)->@location(0) vec4<f32>{
 let c=textureSampleLevel(image,linearSampler,v.uv,0);return vec4<f32>(c.rgb+vec3<f32>(0.025,0.045,0.06)*(1-c.a),1);
}
`;

export async function createSmokeRenderer({device,canvas,positions,velocities,indices,count,radius=.02,slices=32}){
 if(!Number.isInteger(count)||count<1||!Number.isInteger(slices)||slices<1||slices>128||!Number.isFinite(radius)||radius<=0)throw Error('Invalid smoke rendering dimensions.');
 const context=canvas.getContext('webgpu'),format=navigator.gpu.getPreferredCanvasFormat();
 context.configure({device,format,alphaMode:'opaque'});
 const module=device.createShaderModule({code:shader}),screen=device.createShaderModule({code:composite});
 for(const m of [module,screen]){const errors=(await m.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>m.message).join('\n'));}
 const lightCamera=new PerspectiveCamera(45,1,1,200);lightCamera.coordinateSystem=WebGPUCoordinateSystem;lightCamera.position.set(5,5,-5);lightCamera.lookAt(0,0,0);lightCamera.updateProjectionMatrix();lightCamera.updateMatrixWorld();
 const lightVP=new Matrix4().multiplyMatrices(lightCamera.projectionMatrix,lightCamera.matrixWorldInverse);
 const sampler=device.createSampler({magFilter:'linear',minFilter:'linear'});
 const light=device.createTexture({size:[256,256],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
 const uniformBuffers=[0,1].map(()=>device.createBuffer({size:288,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}));
 const blend=(front)=>({color:{srcFactor:front?'one-minus-dst-alpha':'one',dstFactor:front?'one':'one-minus-src-alpha',operation:'add'},alpha:{srcFactor:front?'one-minus-dst-alpha':'one',dstFactor:front?'one':'one-minus-src-alpha',operation:'add'}});
 const make=(fragment,blending)=>device.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:fragment,targets:[{format:'rgba16float',blend:blending}]},primitive:{topology:'triangle-list'}});
 const back=await make('shadowed',blend(false)),front=await make('shadowed',blend(true)),shadow=await make('lightParticle',{color:{srcFactor:'one',dstFactor:'one-minus-src',operation:'add'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'}});
 const display=await device.createRenderPipelineAsync({layout:'auto',vertex:{module:screen,entryPoint:'vertex'},fragment:{module:screen,entryPoint:'fragment',targets:[{format}]},primitive:{topology:'triangle-list'}});
 const bindings=pipeline=>device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[positions,velocities,indices,uniformBuffers[pipeline===shadow?1:0]].map((buffer,binding)=>({binding,resource:{buffer:buffer.gpuBuffer||buffer}}))});
 const groups=new Map([back,front,shadow].map(p=>[p,bindings(p)]));
 const shadowGroups=new Map([back,front].map(p=>[p,device.createBindGroup({layout:p.getBindGroupLayout(1),entries:[{binding:0,resource:light.createView()},{binding:1,resource:sampler}]})]));
 let image,imageView,screenGroup,width=0,height=0,destroyed=false;
 function resize(w,h){if(w===width&&h===height)return;width=w;height=h;image?.destroy();canvas.width=w;canvas.height=h;image=device.createTexture({size:[w,h],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});imageView=image.createView();screenGroup=device.createBindGroup({layout:display.getBindGroupLayout(0),entries:[{binding:0,resource:imageView},{binding:1,resource:sampler}]});}
 function orientation(camera){camera.updateMatrixWorld();const view=camera.getWorldDirection(new Vector3()),lightDirection=lightCamera.position.clone().normalize(),inverted=view.dot(lightDirection)<=0;return {inverted,direction:lightDirection.add(view.multiplyScalar(inverted?-1:1)).normalize()};}
 function write(camera,buffer,shadowAlpha){const data=new Float32Array(72);data.set(camera.matrixWorldInverse.elements,0);data.set(camera.projectionMatrix.elements,16);data.set(camera.matrixWorld.elements,32);data.set(lightVP.elements,48);data.set([radius,.5,.1,shadowAlpha],64);device.queue.writeBuffer(buffer,0,data);}
 function render(camera,{shadows=true}={}){
  if(destroyed)throw Error('Smoke renderer disposed');camera.updateMatrixWorld();const {inverted}=orientation(camera);write(camera,uniformBuffers[0],shadows ? .005 : 0);write(lightCamera,uniformBuffers[1],shadows ? .005 : 0);
  const encoder=device.createCommandEncoder(),lightView=light.createView();
  let pass=encoder.beginRenderPass({colorAttachments:[{view:lightView,loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:.5,a:0}}]});pass.end();
  const pipeline=inverted?front:back;
  for(let slice=0;slice<slices;slice++){
   const begin=Math.floor(count*slice/slices),end=Math.floor(count*(slice+1)/slices);
   pass=encoder.beginRenderPass({colorAttachments:[{view:imageView,loadOp:slice?'load':'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:0}}]});pass.setPipeline(pipeline);pass.setBindGroup(0,groups.get(pipeline));pass.setBindGroup(1,shadowGroups.get(pipeline));pass.draw(6,end-begin,0,begin);pass.end();
   pass=encoder.beginRenderPass({colorAttachments:[{view:lightView,loadOp:'load',storeOp:'store'}]});pass.setPipeline(shadow);pass.setBindGroup(0,groups.get(shadow));pass.draw(6,end-begin,0,begin);pass.end();
  }
  pass=encoder.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});pass.setPipeline(display);pass.setBindGroup(0,screenGroup);pass.draw(3);pass.end();device.queue.submit([encoder.finish()]);
 }
 return {resize,render,orientation,get image(){return image;},light,get destroyed(){return destroyed;},dispose(){destroyed=true;image?.destroy();light.destroy();uniformBuffers.forEach(b=>b.destroy());context.unconfigure();}};
}
