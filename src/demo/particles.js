import * as THREE from 'three/webgpu';
import {instanceIndex,mix,color,float,uv,smoothstep} from 'three/tsl';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createSharedFloat4} from '../runtime/three-bridge.js';
function seedParticles(count){
  const position=new Float32Array(count*4),velocity=new Float32Array(count*4);let state=0x7f4a7c15;
  const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296;};
  for(let i=0;i<count;i++){
    const radius=1.3+Math.sqrt(random())*9.3,theta=random()*Math.PI*2,y=(random()-0.5)*(1.4+0.055*radius),speed=0.55+random()*0.6;
    position.set([Math.cos(theta)*radius,y,Math.sin(theta)*radius,random()],i*4);
    velocity.set([-Math.sin(theta)*speed,(random()-0.5)*0.06,Math.cos(theta)*speed,0],i*4);
  }
  return {position,velocity};
}
export async function createParticleDemo(container,runtime,source,{count=131072,onStats=()=>{},onError=console.error}={}){
  const renderer=new THREE.WebGPURenderer({device:runtime.device,antialias:false,alpha:false});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x080d15,1);renderer.onError=e=>onError(new Error(e.message||String(e)));
  container.appendChild(renderer.domElement);await renderer.init();
  if(!renderer.backend?.isWebGPUBackend)throw new Error('The renderer fell back to WebGL. WebGPU is required.');
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(44,1,0.1,150);camera.position.set(13,11,18);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=0.055;controls.minDistance=5;controls.maxDistance=55;controls.target.set(0,0,0);
  const grid=new THREE.GridHelper(36,36,0x304256,0x192536);grid.position.y=-2.6;grid.material.transparent=true;grid.material.opacity=0.55;scene.add(grid);
  const ringMaterial=new THREE.LineBasicMaterial({color:0x527078,transparent:true,opacity:0.25});const rings=[];
  for(const radius of [3,6,10,14]){const points=Array.from({length:192},(_,i)=>new THREE.Vector3(Math.cos(i/192*Math.PI*2)*radius,-2.58,Math.sin(i/192*Math.PI*2)*radius));const ring=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points),ringMaterial);scene.add(ring);rings.push(ring);}
  let positions,velocities,material,sprites,invocation,kernel,block=128,paused=false,suspended=false,disposed=false,time=0,last=performance.now(),statStart=last,frames=0;
  const create=async(newCount)=>{
    if(!Number.isInteger(newCount)||newCount<1||newCount>1048576)throw new RangeError('Particle count must be in [1,1048576].');
    if(sprites){scene.remove(sprites);material.dispose();positions.dispose();velocities.dispose();}
    count=newCount;const data=seedParticles(count);
    positions=createSharedFloat4(renderer,runtime,data.position,{label:'CUDA positions / Three vertex storage'});
    velocities=createSharedFloat4(renderer,runtime,data.velocity,{label:'CUDA velocity storage'});
    kernel=await runtime.kernel(source,{entry:'particles',workgroupSize:[block,1,1]});
    invocation=kernel.bind({position:positions.resource,velocity:velocities.resource},{n:count,dt:0.016,time:0,attraction:1});
    material=new THREE.SpriteNodeMaterial();const p=positions.node.element(instanceIndex);
    material.positionNode=p.xyz;material.scaleNode=float(count>262144?0.035:0.045);
    material.colorNode=mix(color(0xf0b269),color(0x5ce5d2),p.w);
    material.opacityNode=float(1).sub(smoothstep(float(0.03),float(0.5),uv().sub(0.5).length())).mul(0.6);
    material.transparent=true;material.depthWrite=false;material.depthTest=true;material.blending=THREE.AdditiveBlending;
    sprites=new THREE.Sprite(material);sprites.count=count;sprites.frustumCulled=false;scene.add(sprites);time=0;
  };
  const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(w<1||h<1)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);};
  const observer=new ResizeObserver(resize);observer.observe(container);resize();await create(count);
  const frame=()=>{
    if(disposed)return;const now=performance.now(),dt=Math.min(Math.max((now-last)/1000,0),1/30);last=now;
    if(suspended||document.hidden)return;
    try{
      if(!paused){time+=dt;invocation.setScalars({dt,time});runtime.batch().dispatch(invocation,[Math.ceil(count/block)]).submit();}
      controls.update();renderer.render(scene,camera);frames++;
      if(now-statStart>=600){onStats({fps:frames*1000/(now-statStart),frameMs:(now-statStart)/Math.max(1,frames),count,block,paused,gpuStateBytes:count*32,readbackBytes:runtime.stats.readbackBytes});frames=0;statStart=now;}
    }catch(error){suspended=true;onError(error);}
  };
  renderer.setAnimationLoop(frame);
  return {
    renderer,get positions(){return positions;},get count(){return count;},
    pause(value){paused=!!value;},suspend(value){suspended=!!value;last=performance.now();statStart=last;frames=0;},
    async reset(newCount=count){const old=suspended;suspended=true;await runtime.idle();try{await create(newCount);}finally{suspended=old;last=performance.now();}},
    async applyParticleArtifact(artifact){
      const names=artifact.metadata.bindings.map(b=>b.name).join(','),scalars=artifact.metadata.scalars.map(s=>`${s.name}:${s.type}`).join(',');
      if(artifact.name!=='particles'||names!=='position,velocity'||artifact.metadata.bindings.some(b=>b.elementType!=='vec4<f32>')||scalars!=='n:u32,dt:f32,time:f32,attraction:f32'||artifact.metadata.workgroupSize[1]!==1||artifact.metadata.workgroupSize[2]!==1)throw new Error('Live replacement must preserve the particles kernel buffer/scalar ABI and use a 1D workgroup.');
      const next=await runtime.kernel(artifact);kernel=next;block=artifact.metadata.workgroupSize[0];invocation=kernel.bind({position:positions.resource,velocity:velocities.resource},{n:count,dt:0.016,time,attraction:1});
    },
    async dispose(){if(disposed)return;disposed=true;renderer.setAnimationLoop(null);observer.disconnect();controls.dispose();await runtime.idle();material.dispose();positions.dispose();velocities.dispose();grid.geometry.dispose();grid.material.dispose();rings.forEach(r=>r.geometry.dispose());ringMaterial.dispose();renderer.dispose();renderer.domElement.remove();}
  };
}
