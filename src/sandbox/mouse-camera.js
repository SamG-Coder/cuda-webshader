// Sandbox-only pointer input. All scene rendering still runs the editor CUDA.
export function attachMouseCamera({canvas,runtime,result,spec,owner,checkbox,isCurrent,log}){
 const status=document.createElement('div'),reset=document.getElementById('reset-camera');
 status.className='mouse-camera-status';reset.textContent='Reset camera';reset.type='button';
 canvas.parentElement.prepend(status);reset.hidden=false;canvas.style.touchAction='none';canvas.style.cursor='grab';canvas.tabIndex=0;canvas.setAttribute('aria-label','Path tracer camera: drag to orbit, scroll to zoom');
 let yaw=Math.atan2(3,13),pitch=Math.atan2(2,Math.hypot(13,3)),radius=Math.hypot(13,2,3),pointer=null,dirty=false,disposed=false,completed=0;
 const input=new Float32Array(4),controller=new AbortController(),options={signal:controller.signal};
 const paintStatus=()=>{status.textContent=!checkbox.checked?'Mouse camera off':`Mouse camera test · ${spec.width} × ${spec.height} · drag to orbit, wheel to zoom`;};paintStatus();
 const render=()=>{
  if(disposed||!isCurrent()||!checkbox.checked||owner.framePromise)return;
  owner.framePromise=(async()=>{while(dirty&&!disposed&&isCurrent()&&checkbox.checked){
   dirty=false;runtime.write(result.buffers.mouse_camera,input);status.textContent='Tracing new camera view…';
   await result.stepFrame();if(!isCurrent())break;
   result.inspection=await runtime.read(result.buffers[spec.buffer],Float32Array);
   if(!isCurrent())break;drawPixels(canvas,result.inspection,spec);completed++;paintStatus();
  }})().catch(error=>{dirty=false;status.textContent='Camera render failed: '+error.message;log(status.textContent,'error');}).finally(()=>{owner.framePromise=null;});
 };
 const request=()=>{input.set([radius*Math.cos(pitch)*Math.cos(yaw),radius*Math.sin(pitch),radius*Math.cos(pitch)*Math.sin(yaw),1]);dirty=true;render();};
 canvas.addEventListener('pointerdown',e=>{if(!checkbox.checked||e.button!==0||pointer!==null)return;pointer={id:e.pointerId,x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing';e.preventDefault();},options);
 canvas.addEventListener('pointermove',e=>{if(!checkbox.checked||pointer?.id!==e.pointerId)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;pointer.x=e.clientX;pointer.y=e.clientY;yaw-=dx*.006;pitch=Math.max(-1.35,Math.min(1.35,pitch+dy*.006));request();},options);
 const release=e=>{if(pointer?.id!==e.pointerId)return;pointer=null;canvas.style.cursor='grab';};
 for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,release,options);
 canvas.addEventListener('wheel',e=>{if(!checkbox.checked)return;e.preventDefault();radius=Math.max(3,Math.min(50,radius*Math.exp(Math.max(-200,Math.min(200,e.deltaY))*.001)));request();},{...options,passive:false});
 reset.addEventListener('click',()=>{yaw=Math.atan2(3,13);pitch=Math.atan2(2,Math.hypot(13,3));radius=Math.hypot(13,2,3);input.fill(0);dirty=true;render();},options);
 owner.disposers.push(()=>{disposed=true;controller.abort();reset.hidden=true;reset.disabled=true;});
 owner.mouseCamera={sync(){dirty=false;pointer=null;canvas.style.touchAction=checkbox.checked?'none':'auto';canvas.style.cursor=checkbox.checked?'grab':'default';reset.disabled=!checkbox.checked;paintStatus();}};owner.mouseCamera.sync();
 result.mouseCamera={get completedFrames(){return completed;},get input(){return [...input];}};
 log('Mouse input enabled: the visible sandbox_mouse_camera CUDA kernel receives camera xyz. Each view reruns the editor pipeline; no compiler special case.');
}

function drawPixels(canvas,data,spec){
 const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(spec.width,spec.height);
 for(let y=0;y<spec.height;y++)for(let x=0;x<spec.width;x++){
  const dst=(y*spec.width+x)*4,src=((spec.flipY?spec.height-1-y:y)*spec.width+x)*3;
  for(let c=0;c<3;c++)pixels.data[dst+c]=Math.floor(Math.max(0,Math.min(.999,data[src+c]))*255.99);
  pixels.data[dst+3]=255;
 }ctx.putImageData(pixels,0,0);
}
// Add input plumbing without changing image dimensions, sample count or launch sizes.
export function mouseCameraPlan(original){
 const plan=structuredClone(original);
 if(plan.preview?.kind!=='image'||plan.preview.format!=='rgb-f32')throw Error('Mouse camera requires the path-tracer image pipeline.');
 if(plan.steps.some(s=>s.entry==='sandbox_mouse_camera'))return plan;
 const index=plan.steps.findIndex(s=>s.entry==='create_world');
 if(index<0||!plan.steps.some(s=>s.entry==='render'))throw Error('Path-tracer camera stages are missing.');
 plan.buffers.mouse_camera={type:'f32',records:4,fill:'zero'};
 plan.steps.splice(index+1,0,{entry:'sandbox_mouse_camera',block:[1,1,1],groups:[1,1,1],scalars:{nx:plan.steps[index].scalars.nx,ny:plan.steps[index].scalars.ny}});
 plan.preview.mouseCamera=true;return plan;
}
