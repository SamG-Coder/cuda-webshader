import * as THREE from 'three/webgpu';
import {instanceIndex,positionLocal,mix,color} from 'three/tsl';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {createSharedFloat4} from '../runtime/three-bridge.js';

// Display only: positions and pressure are borrowed by the CUDA/WebGPU solver.
// The renderer never advances particles or uploads simulated state.
export async function createChronoRenderer({runtime,container,positions,properties,fluidCount=16731}){
 if(!(positions instanceof Float32Array)||!(properties instanceof Float32Array)||positions.length!==properties.length||positions.length%4||fluidCount<1||fluidCount>positions.length/4)throw Error('Invalid Chrono render buffers');
 const renderer=new THREE.WebGPURenderer({device:runtime.device,antialias:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x08131d);await renderer.init();container.replaceChildren(renderer.domElement);
 const sharedPos=createSharedFloat4(renderer,runtime,positions,{label:'Chrono original marker positions'}),sharedRho=createSharedFloat4(renderer,runtime,properties,{label:'Chrono original marker properties'});
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.05,100);
 camera.up.set(0,0,1);camera.position.set(10,-17,10);
 const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(-1.5,0,1.8);controls.enableDamping=true;controls.update();
 const material=new THREE.MeshStandardNodeMaterial({roughness:.28,metalness:.12});
 const p=sharedPos.node.element(instanceIndex),rho=sharedRho.node.element(instanceIndex);
 material.positionNode=positionLocal.add(p.xyz);
 material.colorNode=mix(color(0x087bbf),color(0x95edff),rho.y.div(40000).clamp(0,1));
 const geometry=new THREE.SphereGeometry(.055,8,6),water=new THREE.InstancedMesh(geometry,material,fluidCount);water.frustumCulled=false;scene.add(water);
 scene.add(new THREE.HemisphereLight(0xe8f8ff,0x15212e,2.5));const light=new THREE.DirectionalLight(0xffffff,4);light.position.set(-2,-6,12);scene.add(light);
 const floorGeometry=new THREE.PlaneGeometry(12,1.1),floorMaterial=new THREE.MeshStandardMaterial({color:0x253b4b,roughness:.8,side:THREE.DoubleSide}),floor=new THREE.Mesh(floorGeometry,floorMaterial);floor.position.z=-.11;scene.add(floor);
 const frameGeometry=new THREE.EdgesGeometry(new THREE.BoxGeometry(12,1.1,8)),frameMaterial=new THREE.LineBasicMaterial({color:0x41647a,transparent:true,opacity:.45}),frame=new THREE.LineSegments(frameGeometry,frameMaterial);frame.position.z=4;scene.add(frame);
 const resize=()=>{const width=container.clientWidth,height=container.clientHeight;if(!width||!height)return;camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height);};
 const observer=new ResizeObserver(resize);observer.observe(container);resize();
 return {positions:sharedPos.resource,properties:sharedRho.resource,renderer,camera,controls,
  render(){controls.update();renderer.render(scene,camera);},
  dispose(){observer.disconnect();controls.dispose();geometry.dispose();material.dispose();floorGeometry.dispose();floorMaterial.dispose();frameGeometry.dispose();frameMaterial.dispose();sharedPos.dispose();sharedRho.dispose();renderer.dispose();}
 };
}
