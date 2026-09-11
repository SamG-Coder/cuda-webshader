import {GpuRuntime} from './runtime/runtime.js';import {CompilerClient} from './compiler/client.js';
import {KERNELS,loadKernelSources,kernelOptions} from './kernels.js';import {createParticleDemo} from './demo/particles.js';
import {runGpuSuite} from '../tests/gpu-suite.js';import {runThreeInteropTest} from '../tests/three-interop.js';
import {makeCases} from '../tests/cases.js';import {runBenchmarks} from './runtime/benchmark.js';
const $=id=>document.getElementById(id),compiler=new CompilerClient(),sources=await loadKernelSources();
let runtime,demo,artifact=null,artifactValidated=false,busy=false,paused=false,testsReport=null,benchmarkReport=null,compileTicket=0;
const activity=text=>{$('activity').textContent=text;};
function errorMessage(error){const message=error?.message||String(error);activity(message);return message;}
function tab(name){document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===`panel-${name}`));}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.tab));document.querySelectorAll('[data-open-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.openTab));
function download(name,data,type='application/json'){const link=document.createElement('a'),url=URL.createObjectURL(new Blob([typeof data==='string'?data:JSON.stringify(data,null,2)],{type}));link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function setBusy(value,label='GPU work in progress'){
  busy=value;demo?.suspend(value);$('busy-overlay').hidden=!value;$('busy-label').textContent=label;
  for(const id of ['validate','benchmark','pause','reset','particle-count'])$(id).disabled=value||!runtime||!demo;
  $('compile').disabled=value;$('apply-live').disabled=value||!demo||!artifactValidated||!artifact||artifact.name!=='particles';
}
for(const k of KERNELS){const o=document.createElement('option');o.value=k.id;o.textContent=k.title;$('kernel-select').appendChild(o);}
$('kernel-select').value='particles';
function invalidateSource(){compileTicket++;artifact=null;artifactValidated=false;$('export-shader').disabled=true;$('apply-live').disabled=true;$('compile-info').textContent='Modified / uncompiled';}
function selectKernel(){invalidateSource();const id=$('kernel-select').value;if(id==='custom')return;const k=KERNELS.find(k=>k.id===id);$('source-editor').value=sources[id];$('block-shape').value=k.workgroupSize.join(',');artifact=null;$('export-shader').disabled=true;$('apply-live').disabled=true;$('compile-info').textContent='Edited / uncompiled';showEditor(false);}
$('kernel-select').onchange=selectKernel;selectKernel();
function showEditor(wgsl){$('source-editor').hidden=wgsl;$('wgsl-editor').hidden=!wgsl;$('show-cuda').classList.toggle('active',!wgsl);$('show-wgsl').classList.toggle('active',wgsl);}
$('show-cuda').onclick=()=>showEditor(false);$('show-wgsl').onclick=()=>showEditor(true);
$('source-editor').addEventListener('input',invalidateSource);$('block-shape').addEventListener('input',invalidateSource);
async function compileSelected(){
  const ticket=++compileTicket;artifactValidated=false;$('compiler-message').classList.remove('error');$('compiler-message').textContent='Compiling in worker…';$('apply-live').disabled=true;
  try{
    const block=$('block-shape').value.split(',').map(x=>Number(x.trim())),id=$('kernel-select').value;
    const result=await compiler.compile($('source-editor').value,{entry:id==='custom'?undefined:id,workgroupSize:block});if(ticket!==compileTicket)return;
    artifact=result.artifact;$('wgsl-editor').value=artifact.wgsl;$('export-shader').disabled=false;$('compile-info').textContent=`${result.compileMs.toFixed(1)} ms CPU`;
    if(runtime)await runtime.kernel(artifact);if(ticket!==compileTicket)return;artifactValidated=!!runtime;
    const m=artifact.metadata;$('compiler-message').textContent=`${runtime?'WGSL accepted by this GPU.':'Generated WGSL — GPU validation unavailable.'}\n${m.bindings.length} storage bindings · ${m.uniformSize} uniform bytes\nWorkgroup ${m.workgroupSize.join(' × ')} · ${m.workgroupStorageBytes} shared bytes\nBarrier scope: ${m.barrier}`;
    $('apply-live').disabled=busy||!demo||!artifactValidated||artifact.name!=='particles';activity(`${artifact.name}: ${runtime?'compiled and GPU-validated':'source translated only'}.`);
  }catch(error){$('compiler-message').classList.add('error');$('compiler-message').textContent=errorMessage(error);}
}
$('compile').onclick=compileSelected;
$('export-shader').onclick=()=>artifact&&download(`${artifact.name}.wgsl`,artifact.wgsl,'text/plain');
$('open-source').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;invalidateSource();if(file.size>1_000_000)throw new Error('Source exceeds the 1 MB compiler limit.');if(!$('kernel-select').querySelector('[value="custom"]')){const o=document.createElement('option');o.value='custom';o.textContent='Custom CUDA source';$('kernel-select').appendChild(o);}$('kernel-select').value='custom';$('source-editor').value=await file.text();artifact=null;showEditor(false);$('compile-info').textContent='Custom / uncompiled';$('export-shader').disabled=true;$('apply-live').disabled=true;activity(`Opened ${file.name}.`);}catch(error){errorMessage(error);}};
$('apply-live').onclick=async()=>{try{if(!artifact||!artifactValidated||!demo)return;await demo.applyParticleArtifact(artifact);activity('Live particle kernel replaced. GPU state was preserved.');}catch(error){$('compiler-message').classList.add('error');$('compiler-message').textContent=errorMessage(error);}};
$('pause').onclick=()=>{paused=!paused;demo.pause(paused);$('pause').textContent=paused?'Resume simulation':'Pause simulation';$('simulation-state').textContent=paused?'PAUSED':'RUNNING';};
async function reset(){if(!demo||busy)return;setBusy(true,'Recreating particle state');try{await demo.reset(Number($('particle-count').value));activity(`Particle buffers reset to ${demo.count.toLocaleString()} records.`);}catch(error){errorMessage(error);}finally{setBusy(false);}}
$('reset').onclick=reset;$('particle-count').onchange=reset;
function addTest(result){
  const row=document.createElement('div');row.className=`test-row${result.pass?'':' fail'}`;const badge=document.createElement('b');badge.textContent=result.pass?'PASS':'FAIL';const text=document.createElement('span');text.textContent=result.name;row.append(badge,text);
  if(!result.pass){const error=document.createElement('small');error.textContent=result.error;text.appendChild(error);}$('test-results').appendChild(row);row.scrollIntoView({block:'nearest'});
}
async function runTests(){
  if(!runtime||busy)return;tab('validate');setBusy(true,'Validating translated kernels');$('test-results').replaceChildren();testsReport=null;$('export-tests').disabled=true;
  let passed=0,failed=0;const expectedTotal=makeCases().length+5;
  const update=result=>{result.pass?passed++:failed++;addTest(result);$('test-passed').textContent=passed;$('test-failed').textContent=failed;$('test-total').textContent=passed+failed;$('test-progress').style.width=`${(passed+failed)/expectedTotal*100}%`;activity(`${passed+failed}/${expectedTotal} GPU tests: ${passed} passed, ${failed} failed.`);};
  try{
    await runtime.idle();testsReport=await runGpuSuite(runtime,sources,{onCase:update});let interop;
    try{interop=await runThreeInteropTest(runtime);}catch(error){interop={name:'Three r186 shared-buffer rendered-pixel test',pass:false,error:String(error.stack||error)};}
    testsReport.results.push(interop);testsReport.total++;interop.pass?testsReport.passed++:testsReport.failed++;update(interop);
    $('export-tests').disabled=false;activity(`GPU test suite complete: ${testsReport.passed}/${testsReport.total} passed. ${testsReport.failed} failed.`);
  }catch(error){errorMessage(error);}finally{setBusy(false);}
  return testsReport;
}
$('validate').onclick=runTests;$('export-tests').onclick=()=>testsReport&&download('webgpu-correctness.json',testsReport);
function addBenchmark(result){
  const row=document.createElement('div');row.className='bench-row';const top=document.createElement('div'),name=document.createElement('span'),value=document.createElement('strong');name.textContent=result.name;value.textContent=`${result.medianMs.toFixed(4)} ms`;top.append(name,value);
  const note=document.createElement('small');note.textContent=`${result.metric.startsWith('GPU')?'GPU timestamp':'Wall time'} · ${result.gflops.toFixed(2)} GFLOP/s${result.effectiveGBs?` · ${result.effectiveGBs.toFixed(2)} logical GB/s`:''}`;row.append(top,note);$('benchmark-results').appendChild(row);
  activity(`${result.name}: measured ${result.medianMs.toFixed(4)} ms per dispatch.`);
}
async function benchmark(){
  if(!runtime||busy)return;tab('bench');setBusy(true,'Measuring kernel variants');$('benchmark-results').replaceChildren();$('tuning-winners').hidden=true;$('export-bench').disabled=true;
  try{
    await runtime.idle();benchmarkReport=await runBenchmarks(runtime,sources,{n:Number($('benchmark-n').value),matrixSize:Number($('benchmark-m').value),onResult:addBenchmark});
    const box=$('tuning-winners');box.replaceChildren();const heading=document.createElement('b');heading.textContent='Measured winners on this device';box.append(heading);
    for(const [family,choice]of Object.entries(benchmarkReport.choices)){const p=document.createElement('p');p.textContent=`${family}: ${choice.id}, block ${choice.workgroupSize.join('×')}${choice.baselineSpeedup?` · ${choice.baselineSpeedup.toFixed(2)}× against naive WebGPU` : ''}`;box.append(p);}box.hidden=false;$('export-bench').disabled=false;activity('Benchmarks complete. Export includes raw samples, device details and selected variants.');
  }catch(error){errorMessage(error);const row=document.createElement('p');row.textContent=error.message;row.style.color='var(--red)';$('benchmark-results').append(row);}finally{setBusy(false);}
  return benchmarkReport;
}
$('benchmark').onclick=benchmark;$('export-bench').onclick=()=>benchmarkReport&&download('webgpu-benchmark.json',benchmarkReport);
try{
  runtime=await GpuRuntime.create({onError:error=>{$('fatal').hidden=false;$('fatal').textContent=errorMessage(error);}});
  const info=runtime.describe();$('device-status').textContent=`WEBGPU / ${(info.description||info.vendor).slice(0,45)}`;$('status-dot').classList.add('ready');
  $('timer-note').textContent=info.timestampQuery?'GPU timestamp queries available. Raw batch samples and wall times are included.':'GPU timestamps unavailable. Timings will be explicitly labeled submit-to-completion wall time.';
  demo=await createParticleDemo($('viewport'),runtime,sources.particles,{onStats:s=>{$('hud-count').textContent=s.count.toLocaleString();$('hud-fps').innerHTML=`${s.fps.toFixed(0)} <small>FPS</small>`;$('hud-memory').innerHTML=`${(s.gpuStateBytes/1048576).toFixed(1)} <small>MiB</small>`;},onError:error=>{$('fatal').hidden=false;$('fatal').textContent=errorMessage(error);}});
  $('simulation-state').textContent='RUNNING';setBusy(false);activity('CUDA particle kernel running. Position and velocity buffers stay on the GPU.');
}catch(error){$('device-status').textContent='GPU UNAVAILABLE';$('fatal').hidden=false;$('fatal').textContent=errorMessage(error)+'\nThe source compiler remains available in Kernel lab. No CPU simulation is substituted.';$('simulation-state').textContent='UNAVAILABLE';}
await compileSelected();
window.cudaWebShader={runtime,demo,sources,runTests,benchmark};
window.addEventListener('beforeunload',()=>compiler.dispose());
