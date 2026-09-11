import * as THREE from 'three/webgpu';
import {instanceIndex,float,color,mix} from 'three/tsl';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {CompilerClient} from '../compiler/client.js';
import {GpuRuntime} from '../runtime/runtime.js';
import {createSharedFloat4} from '../runtime/three-bridge.js';
import {suggestLaunch,suggestConfig,validateConfig,seedBuffer} from './config.js';
import {kernelSource} from './import.js';
import {createShaderView} from './shader-view.js';
const $=id=>document.getElementById(id),compiler=new CompilerClient();
let editor,runtime,shaderView,active=null,busy=false,configKey='',lastArtifact=null,dirty=false,runCount=0,dropDepth=0;
function log(message,level='info'){const row=document.createElement('div');row.className=`log-row log-${level}`;const stamp=document.createElement('span');stamp.className='log-time';stamp.textContent=new Date().toLocaleTimeString('en-GB');const text=document.createElement('span');text.textContent=message;row.append(stamp,text);$('log').append(row);while($('log').children.length>200)$('log').firstChild.remove();$('log').scrollTop=$('log').scrollHeight;}
function state(text){$('state').textContent=text;}
function stop(reason='Stopped. GPU output preserved.'){if(active)active.running=false;$('stop').disabled=true;if(reason){state(reason);log(reason);}}
async function cleanup(){if(!active)return;const old=active;active=null;old.renderer?.setAnimationLoop(null);old.observer?.disconnect();await runtime.idle();old.controls?.dispose();old.material?.dispose();old.shared?.dispose();for(const r of old.resources)runtime.destroyBuffer(r);old.renderer?.dispose();$('preview').replaceChildren();}
function download(name,text){const url=URL.createObjectURL(new Blob([text],{type:'text/plain'})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function detect(){const suggestion=suggestLaunch(editor.getValue());$('entry').value=suggestion.entry;$('block').value=suggestion.block.join(',');configKey='';log(`Detected entry ${suggestion.entry||'(none)'}; suggested block ${suggestion.block.join(' × ')}. Inspect launch settings for custom kernels.`);}
function showError(error){const message=error.message||String(error);log(message,'error');state('Failed — see execution log');const match=message.match(/\((\d+):(\d+)\)/);if(editor&&match){const line=Number(match[1]),col=Number(match[2]);monaco.editor.setModelMarkers(editor.getModel(),'cuda',[{startLineNumber:line,startColumn:col,endLineNumber:line,endColumn:col+1,message,severity:monaco.MarkerSeverity.Error}]);editor.revealLineInCenter(line);}window.sandbox.lastError=message;}
function numericPreview(data,name){const wrapper=document.createElement('div');wrapper.className='numeric';const heading=document.createElement('h3');heading.textContent=`${name} · ${data.length.toLocaleString()} values`;wrapper.append(heading);const canvas=document.createElement('canvas');canvas.width=256;canvas.height=64;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(256,64);let min=Infinity,max=-Infinity;for(const v of data){if(Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);}}for(let i=0;i<256*64;i++){const value=data[Math.min(data.length-1,Math.floor(i/(256*64)*data.length))],t=Number.isFinite(value)?(value-min)/(max-min||1):0;pixels.data.set([30+Math.round(t*100),60+Math.round(t*180),100+Math.round(t*120),255],i*4);}ctx.putImageData(pixels,0,0);wrapper.append(canvas);const note=document.createElement('p');note.textContent=`Range ${min} … ${max}. Heatmap samples the output; first 128 values below.`;wrapper.append(note);const table=document.createElement('table');for(let i=0;i<Math.min(128,data.length);i++){const row=table.insertRow();row.insertCell().textContent=String(i);row.insertCell().textContent=String(data[i]);}wrapper.append(table);$('preview').replaceChildren(wrapper);}
async function run(){
 if(busy)return;busy=true;dirty=false;runCount++;window.sandbox.lastError=null;stop(null);$('run').disabled=true;const revision=editor.getModel().getVersionId();
 try{
  await cleanup();monaco.editor.setModelMarkers(editor.getModel(),'cuda',[]);state('Compiling…');const source=editor.getValue();if(new TextEncoder().encode(source).length>1000000)throw Error('Source exceeds the 1 MB limit.');
  const block=$('block').value.split(',').map(Number);if(block.length!==3)throw Error('Threads per block needs three comma-separated dimensions.');
  log(`Run ${runCount}: parsing ${source.length.toLocaleString()} characters in compiler worker.`);
  const input=kernelSource(source);
  if(input.extracted)log(`Desktop CUDA file: extracted ${input.functions} device/kernel functions. Includes and host code are not executed; launch settings provide synthetic inputs.`);
  const compiled=await compiler.compile(input.source,{entry:$('entry').value.trim()||undefined,workgroupSize:block});
  if(editor.getModel().getVersionId()!==revision)throw Error('Source changed while compiling. Run again to execute the latest version.');
  const artifact=compiled.artifact;lastArtifact=artifact;shaderView.generated(artifact);log(`${artifact.name}: typed CUDA → WGSL in ${compiled.compileMs.toFixed(2)} ms. ${artifact.metadata.bindings.length} buffer bindings; ${artifact.metadata.workgroupStorageBytes} shared bytes.`,'success');
  const signature=JSON.stringify([artifact.name,artifact.metadata]);
  if(configKey!==signature){$('config').value=JSON.stringify(suggestConfig(artifact),null,2);configKey=signature;log('Generated suggested scalar values, synthetic inputs and launch dimensions. Expand Launch settings to change them.');}
  const config=JSON.parse($('config').value),bytes=validateConfig(config,artifact.metadata);
  runtime ||= await GpuRuntime.create({onError:e=>{stop(null);showError(e);}});const info=runtime.describe();$('device').textContent=`WEBGPU / ${info.vendor.toUpperCase()}`;
  log('Validating WGSL and creating the WebGPU pipeline…');const kernel=await runtime.kernel(artifact);shaderView.validated();log('GPU shader validation passed.','success');
  const selected=artifact.metadata.bindings.find(b=>b.name===config.output),points=selected.elementType==='vec4<f32>';
  const buffers={},resources=[];active={resources,running:false,config,kernel};
  if(points){active.renderer=new THREE.WebGPURenderer({device:runtime.device,antialias:false});active.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));active.renderer.setClearColor(0x080f16);$('preview').replaceChildren(active.renderer.domElement);await active.renderer.init();}
  for(const binding of artifact.metadata.bindings){const spec=config.buffers[binding.name],data=seedBuffer(binding,spec);if(points&&binding.name===config.output){active.shared=createSharedFloat4(active.renderer,runtime,data);buffers[binding.name]=active.shared.resource;}else{buffers[binding.name]=runtime.createBuffer(data);resources.push(buffers[binding.name]);}log(`Buffer ${binding.name}: ${spec.records.toLocaleString()} × ${binding.elementType}, ${spec.fill} fill${binding.readOnly?' (read only)':''}.`);}
  active.invocation=kernel.bind(buffers,config.scalars);active.buffers=buffers;
  log(`Allocated ${(bytes/1048576).toFixed(2)} MiB. Dispatch ${config.groups.join(' × ')} blocks × ${block.join(' × ')} threads.`);state('Running on GPU…');
  const start=performance.now();runtime.batch().dispatch(active.invocation,config.groups).submit();await runtime.idle();const elapsed=performance.now()-start;
  log(`Dispatch completed in ${elapsed.toFixed(2)} ms (CPU submit-to-completion wall time).`,'success');
  const Type=selected.elementType.includes('u32')?Uint32Array:selected.elementType.includes('i32')?Int32Array:Float32Array;
  const data=await runtime.read(buffers[config.output],Type);window.sandbox.lastOutput=Array.from(data.subarray(0,128));window.sandbox.lastConfig=config;
  let nonFinite=0;for(const value of data)if(!Number.isFinite(value))nonFinite++;
  log(`Read ${config.output}: ${data.length.toLocaleString()} components${nonFinite?`, ${nonFinite} non-finite values`:`, all finite`}. This is output inspection, not an algorithm correctness test.`,nonFinite?'error':'info');
  if(points)setupPoints(data,selected,config);else{numericPreview(data,config.output);$('animate').checked=false;log('Numeric preview ready. Run again to recompute with changed inputs.');}
  dirty=editor.getModel().getVersionId()!==revision;
  $('preview-info').textContent=points?`${config.buffers[config.output].records.toLocaleString()} points · ${config.output}.xyz · shared GPU buffer · drag to orbit`:`${config.output} · ${selected.elementType} · ${config.buffers[config.output].records.toLocaleString()} records`;
  if(dirty)stop('Source changed during execution. Preview shows the previous run.');else{state('Run complete');log('Preview ready.','success');}
  window.sandbox.completedRuns=runCount;
 }catch(error){await cleanup().catch(()=>{});shaderView?.failed();showError(error);}finally{busy=false;$('run').disabled=false;}
}
function setupPoints(data,selected,config){
 const a=active,scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1,0.01,100000),bounds=new THREE.Box3();
 for(let i=0;i<data.length;i+=4)if(Number.isFinite(data[i]+data[i+1]+data[i+2]))bounds.expandByPoint(new THREE.Vector3(data[i],data[i+1],data[i+2]));
 const center=bounds.isEmpty()?new THREE.Vector3():bounds.getCenter(new THREE.Vector3()),extent=bounds.isEmpty()?1:Math.max(1,bounds.getSize(new THREE.Vector3()).length());
 camera.position.copy(center).add(new THREE.Vector3(extent*0.75,extent*0.5,extent*0.9));camera.far=Math.max(100,extent*20);camera.updateProjectionMatrix();a.controls=new OrbitControls(camera,a.renderer.domElement);a.controls.target.copy(center);a.controls.enableDamping=true;
 const material=a.material=new THREE.SpriteNodeMaterial(),p=a.shared.node.element(instanceIndex);material.positionNode=p.xyz;material.scaleNode=float(Math.max(extent/Math.sqrt(config.buffers[config.output].records)*0.65,extent*0.0005));material.colorNode=mix(color(0x348ded),color(0xa2f5d3),p.y.sub(center.y).div(extent).add(0.5));material.depthWrite=true;const sprite=new THREE.Sprite(material);sprite.count=config.buffers[config.output].records;sprite.frustumCulled=false;scene.add(sprite);
 const resize=()=>{const w=$('preview').clientWidth,h=$('preview').clientHeight;if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();a.renderer.setSize(w,h);};a.observer=new ResizeObserver(resize);a.observer.observe($('preview'));resize();
 let previous=performance.now(),time=config.scalars.time??0;
 a.running=$('animate').checked&&!dirty;$('stop').disabled=!a.running;
 a.renderer.setAnimationLoop(()=>{if(active!==a)return;const now=performance.now(),dt=Math.min((now-previous)/1000,0.033);previous=now;if(document.hidden)return;try{if(a.running&&!busy&&!dirty){const values={};if('time'in config.scalars){time+=dt;values.time=time;}if('dt'in config.scalars)values.dt=dt;a.invocation.setScalars(values);runtime.batch().dispatch(a.invocation,config.groups).submit();}a.controls.update();a.renderer.render(scene,camera);}catch(e){a.renderer.setAnimationLoop(null);showError(e);}});
}
async function loadFile(file){if(!file)return;if(!/\.(cu|cuh|txt)$/i.test(file.name))throw Error('Choose a .cu, .cuh or .txt source file.');if(file.size>1000000)throw Error('File exceeds 1 MB.');if(busy)throw Error('Wait for the current run before loading a file.');editor.setValue(await file.text());$('filename').textContent=file.name;detect();log(`Loaded ${file.name} locally. Source is not uploaded.`);await run();}
async function example(name){const paths={wave:'showcases/simplegl/kernel.cu',particles:'kernels/particles.cu',saxpy:'kernels/saxpy.cu'};let source;if(paths[name]){const response=await fetch(paths[name]);if(!response.ok)throw Error('Example could not be loaded.');source=await response.text();}else source='__global__ void fill(float* output, unsigned int n) {\n    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) output[i] = (float)i;\n}\n';editor.setValue(source);await editor.getAction('editor.foldAllBlockComments')?.run();$('filename').textContent=paths[name]?.split('/').pop()||'untitled.cu';detect();$('animate').checked=name==='wave'||name==='particles';await run();}
window.sandbox={run,loadFile,get editor(){return editor;},get shaderView(){return shaderView;},get runtime(){return runtime;},completedRuns:0,lastError:null};
try{
 const vs=new URL('../../node_modules/monaco-editor/min/vs',import.meta.url).href;window.require.config({paths:{vs}});
 await new Promise((resolve,reject)=>window.require(['vs/editor/editor.main'],resolve,reject));
 monaco.editor.defineTheme('cuda-dark',{base:'vs-dark',inherit:true,rules:[],colors:{'editor.background':'#0b1520','editorLineNumber.foreground':'#486379','editor.lineHighlightBackground':'#132331'}});
 editor=monaco.editor.create($('editor'),{value:'',language:'cpp',theme:'cuda-dark',automaticLayout:true,fontSize:13,lineHeight:21,minimap:{enabled:false},scrollBeyondLastLine:false,padding:{top:16},tabSize:4,wordWrap:'off',ariaLabel:'CUDA source code',stickyScroll:{enabled:false}});
 shaderView=createShaderView(editor,{download});
 editor.onDidChangeModelContent(()=>{dirty=true;shaderView.markStale();stop(null);state('Source modified — Ctrl/Cmd + Enter to run');monaco.editor.setModelMarkers(editor.getModel(),'cuda',[]);});
 for(const id of ['entry','block'])$(id).addEventListener('input',()=>{shaderView.markStale();stop(null);});
 editor.onDidPaste(()=>{detect();run();});editor.addAction({id:'cuda-run',label:'Compile and run CUDA',keybindings:[monaco.KeyMod.CtrlCmd|monaco.KeyCode.Enter],run});
 $('run').onclick=run;$('stop').onclick=()=>stop();$('infer').onclick=detect;$('open').onclick=()=>$('file').click();$('file').onchange=e=>loadFile(e.target.files[0]).catch(showError);$('export').onclick=()=>download($('filename').textContent,editor.getValue());$('example').onchange=e=>example(e.target.value).catch(showError);$('clear-log').onclick=()=>$('log').replaceChildren();$('reset-config').onclick=()=>{if(lastArtifact){$('config').value=JSON.stringify(suggestConfig(lastArtifact),null,2);log('Suggested inputs restored. Press Compile & run to apply.');}};
 $('animate').onchange=()=>{if(active?.renderer&&!dirty){active.running=$('animate').checked;$('stop').disabled=!active.running;state(active.running?'Animating on GPU':'Paused');}else if($('animate').checked)log('Run a float4 output kernel to enable animated 3D preview.');};
 window.addEventListener('dragenter',e=>{if(e.dataTransfer?.types.includes('Files')){e.preventDefault();dropDepth++;$('drop-overlay').hidden=false;}});window.addEventListener('dragover',e=>{if(e.dataTransfer?.types.includes('Files'))e.preventDefault();});window.addEventListener('dragleave',()=>{if(--dropDepth<=0){dropDepth=0;$('drop-overlay').hidden=true;}});window.addEventListener('drop',e=>{e.preventDefault();dropDepth=0;$('drop-overlay').hidden=true;loadFile(e.dataTransfer?.files[0]).catch(showError);});
 addEventListener('beforeunload',()=>{compiler.dispose();active?.renderer?.setAnimationLoop(null);runtime?.dispose();});log('Monaco editor ready. C++ highlighting, bracket matching, find/replace and compiler error markers enabled.');await example('wave');
}catch(error){showError(error);}
