import * as THREE from 'three/webgpu';
import {instanceIndex,float,color,mix,uv,smoothstep} from 'three/tsl';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GpuRuntime} from '../../src/runtime/runtime.js';
import {createSharedFloat4} from '../../src/runtime/three-bridge.js';
import {compareArrays} from '../../tests/cases.js';
const $=id=>document.getElementById(id);
try {
 const source=await (await fetch('./kernel.cu')).text(),runtime=await GpuRuntime.create(),kernel=await runtime.kernel(source,{entry:'simple_vbo_kernel',workgroupSize:[8,8,1]});
 const renderer=new THREE.WebGPURenderer({device:runtime.device,antialias:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x071016);$('viewport').appendChild(renderer.domElement);await renderer.init();
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(43,1,0.01,100);camera.position.set(2.2,1.8,2.6);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=1;controls.maxDistance=10;
 let shared,material,sprites,invocation,size=256,time=0,paused=false,busy=false,last=performance.now(),frameStart=last,frames=0;
 const resize=()=>{renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();};addEventListener('resize',resize);resize();
 function create(n){if(![128,256,512,1024].includes(n))throw new Error('Grid must be a supported multiple of 8: upstream kernel has no bounds guard.');if(shared){scene.remove(sprites);material.dispose();shared.dispose();}size=n;shared=createSharedFloat4(renderer,runtime,new Float32Array(n*n*4));invocation=kernel.bind({pos:shared.resource},{width:n,height:n,time});material=new THREE.SpriteNodeMaterial();const p=shared.node.element(instanceIndex);material.positionNode=p.xyz;material.scaleNode=float(2.5/n);material.colorNode=mix(color(0x2074eb),color(0x8cffcd),p.y.add(0.5));material.opacityNode=float(1).sub(smoothstep(float(0.15),float(0.5),uv().sub(0.5).length()));material.transparent=true;material.depthWrite=false;sprites=new THREE.Sprite(material);sprites.count=n*n;sprites.frustumCulled=false;scene.add(sprites);}
 function encode(t){invocation.setScalars({time:t});runtime.batch().dispatch(invocation,[size/8,size/8]).submit();}
 create(size);
 async function verify(){busy=true;try{await runtime.idle();const checks=[];for(const t of [0,1.25,7.5]){encode(t);const actual=await runtime.read(shared.resource),expected=new Float32Array(size*size*4);for(let y=0;y<size;y++)for(let x=0;x<size;x++){const u=x/size*2-1,v=y/size*2-1;expected.set([u,Math.sin(u*4+t)*Math.cos(v*4+t)*0.5,v,1],(y*size+x)*4);}const check=compareArrays(actual,expected,{absolute:0.000003,relative:0.000003});checks.push({time:t,...check});if(!check.pass)throw new Error(JSON.stringify(check));}const report={passed:true,grid:[size,size],vertices:size*size,device:runtime.describe(),checks};$('status').textContent=`PASS · ${size*size} vertices × 3 times`;window.showcase.report=report;return report;}finally{encode(time);busy=false;}}
 window.showcase={runtime,renderer,kernel,verify,get size(){return size;},get time(){return time;}};
 await verify();for(const id of ['pause','size','verify'])$(id).disabled=false;
 $('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Resume':'Pause';};
 $('size').onchange=async()=>{busy=true;try{await runtime.idle();create(Number($('size').value));$('status').textContent=`${size*size} GPU vertices`;}finally{busy=false;}};
 $('verify').onclick=()=>verify().catch(fail);
 renderer.setAnimationLoop(()=>{const now=performance.now(),dt=Math.min((now-last)/1000,0.05);last=now;if(busy||document.hidden)return;try{if(!paused){time+=dt;encode(time);}controls.update();renderer.render(scene,camera);frames++;if(now-frameStart>700){$('fps').textContent=`${(frames*1000/(now-frameStart)).toFixed(0)} FPS · ${size*size} points`;frameStart=now;frames=0;}}catch(e){renderer.setAnimationLoop(null);fail(e);}});
}catch(e){fail(e);}
function fail(e){$('error').hidden=false;$('error').textContent=e.stack||String(e);window.showcaseError=String(e);}
