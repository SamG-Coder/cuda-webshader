import * as THREE from 'three/webgpu';
import {float,vertexIndex,instanceIndex,positionLocal,color,mix,transformNormalToView} from 'three/tsl';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createSharedFloat4} from '../runtime/three-bridge.js';
import {executePipeline} from './pipeline.js';

// The renderer borrows the compute output. CPU inspection sets the camera only;
// it never calculates or uploads mesh positions or normals after dispatch.
export async function meshPreview({owner,container,plan,runtime,...options}){
 const particles=plan.preview.kind==='particles',points=particles&&plan.preview.glyph==='points';const renderer=owner.renderer=new THREE.WebGPURenderer({device:runtime.device,antialias:false});
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x080f16);container.replaceChildren(renderer.domElement);await renderer.init();
 const before=runtime.stats.readbackBytes;
 const result=await executePipeline({...options,plan,runtime,resources:owner.resources,textureResources:owner.textureResources,allocateBuffer(name,spec,data){
  if(name===plan.preview.positions||name===plan.preview.normals){const shared=createSharedFloat4(renderer,runtime,data);owner.meshShared.push(shared);if(name===plan.preview.positions)owner.meshPosition=shared;if(name===plan.preview.normals)owner.meshNormal=shared;return shared.resource;}
 }});
 result.controlReadbackBytes=runtime.stats.readbackBytes-before;
 const data=result.count?await runtime.read(result.buffers[plan.preview.positions],Float32Array,result.count*16):new Float32Array();
 const bounds=new THREE.Box3();for(let i=0;i<data.length;i+=4){if(!Number.isFinite(data[i]+data[i+1]+data[i+2]))throw Error('Mesh contains non-finite positions.');bounds.expandByPoint(new THREE.Vector3(data[i],data[i+1],data[i+2]));}
 if(particles&&!points){bounds.expandByPoint(new THREE.Vector3(-1,-1,-1));bounds.expandByPoint(new THREE.Vector3(1,1,1));}
 const center=bounds.isEmpty()?new THREE.Vector3():bounds.getCenter(new THREE.Vector3()),extent=bounds.isEmpty()?2:Math.max(.01,bounds.getSize(new THREE.Vector3()).length());
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,extent/1000,extent*20);
 camera.position.copy(center).add(new THREE.Vector3(.75,.5,.9).multiplyScalar(extent*(particles?.65:1)));
 const material=owner.material=(points?new THREE.SpriteNodeMaterial():new THREE.MeshStandardNodeMaterial({color:Number.isInteger(plan.preview.color)?plan.preview.color:0x85e8cf,roughness:.45,metalness:.1,side:THREE.DoubleSide}));
 let geometry,mesh;
 if(points){const point=owner.meshPosition.node.element(instanceIndex);material.positionNode=point.xyz;material.scaleNode=float(plan.preview.radius*2);material.colorNode=mix(color(0x278bdd),color(0x9af4cc),point.y.sub(center.y).div(extent).add(.5).clamp(0,1));mesh=new THREE.Sprite(material);mesh.count=result.count;}
 else if(particles){const point=owner.meshPosition.node.element(instanceIndex);material.positionNode=positionLocal.add(point.xyz);material.colorNode=mix(color(0x278bdd),color(0x9af4cc),point.y.add(1).div(2).clamp(0,1));geometry=owner.geometry=new THREE.SphereGeometry(plan.preview.radius,8,6);mesh=new THREE.InstancedMesh(geometry,material,result.count);const floor=new THREE.GridHelper(2,16,0x375c6b,0x17303b);floor.position.y=-1;scene.add(floor);owner.floor=floor;}
 else{material.positionNode=owner.meshPosition.node.element(vertexIndex).xyz;material.normalNode=transformNormalToView(owner.meshNormal.node.element(vertexIndex).xyz).toVarying('cw_mesh_normal').normalize();geometry=owner.geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(Math.max(3,result.count)*3),3));geometry.setDrawRange(0,result.count);mesh=new THREE.Mesh(geometry,material);}
 mesh.frustumCulled=false;scene.add(mesh,new THREE.HemisphereLight(0xd8edff,0x182235,2));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,5,4);scene.add(light);
 const controls=owner.controls=new OrbitControls(camera,renderer.domElement);controls.target.copy(center);controls.enableDamping=true;
 const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;camera.aspect=w/h;if(points){const half=THREE.MathUtils.degToRad(camera.fov/2),angle=Math.min(half,Math.atan(Math.tan(half)*camera.aspect)),direction=camera.position.clone().sub(center).normalize();camera.position.copy(center).addScaledVector(direction,extent*.55/Math.sin(angle));}camera.updateProjectionMatrix();renderer.setSize(w,h);};owner.observer=new ResizeObserver(resize);owner.observer.observe(container);resize();
 let simulationTime=0,last=performance.now();renderer.setAnimationLoop(()=>{const now=performance.now(),dt=Math.min(.05,(now-last)/1000);last=now;if(owner.running&&plan.animate)simulationTime+=dt;if(owner.running&&plan.animate&&!owner.framePromise){owner.framePromise=result.stepFrame(simulationTime).then(()=>{owner.animationFrames=(owner.animationFrames||0)+1;}).catch(error=>{owner.running=false;options.log?.(error.message,'error');}).finally(()=>{owner.framePromise=null;});}controls.update();renderer.render(scene,camera);});
 return {...result,get simulationSteps(){return result.steps;},get animationFrames(){return owner.animationFrames||0;},inspection:data,settle:async()=>{await owner.framePromise;}};
}
