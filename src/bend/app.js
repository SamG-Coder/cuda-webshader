import {examples} from './examples.js';
import {GpuRuntime} from '../runtime/runtime.js';
import {runBend,runtimeErrors} from './runtime.js';
const $=id=>document.getElementById(id);
let compiled=null,view='cuda',runtime=null,worker=null,sequence=0;
for(const [key,sample] of Object.entries(examples)) $('example').add(new Option(sample.title,key));
function loadExample(){const e=examples[$('example').value];$('source').value=e.source;$('entry').value=e.entry;$('inputs').value=e.input;invalidate();}
function invalidate(){compiled=null;$('generated').textContent='Compile to inspect the generated runtime and checked program.';$('status').textContent='Source changed. Compile to check this version.';$('status').className='';$('results').replaceChildren();}
function showCode(){if(compiled)$('generated').textContent=view==='cuda'?compiled.cuda:compiled.artifact.wgsl;}
function compile(){
  worker?.terminate();worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  const id=++sequence;
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{worker.terminate();reject(Error('Bend checking exceeded 15 seconds. Simplify the source and retry.'));},15000);
    worker.onmessage=({data})=>{if(data.id!==id)return;clearTimeout(timer);worker.terminate();data.error?reject(Error(data.error)):resolve(data.result);};
    worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};
    worker.postMessage({id,source:$('source').value,entry:$('entry').value});
  });
}
function table(report){
  const table=document.createElement('table'), head=table.createTHead().insertRow();
  for(const title of ['Row','Output','Status','Heap words','Steps']){const th=document.createElement('th');th.textContent=title;head.append(th);}
  report.output.forEach((v,i)=>{const row=table.insertRow();for(const text of [i+1,report.status[i]?'—':v,runtimeErrors[report.status[i]],report.usage[i*2],report.usage[i*2+1]])row.insertCell().textContent=String(text);});
  $('results').replaceChildren(table);
}
async function execute(gpu){
  delete window.__bendLastRun;delete window.__bendLastError;
  const controls=[...document.querySelectorAll('button,input,select,textarea')];controls.forEach(c=>c.disabled=true);
  compiled=null;$('results').replaceChildren();$('generated').textContent='Compiling…';$('status').className='';$('status').textContent='Checking Bend and compiling the CUDA runtime…';
  try {
    compiled=await compile();showCode();
    $('status').textContent=`Bend source checked. CUDA → WGSL compiled.\n${compiled.program.nodes.length} runtime nodes; ${compiled.program.parameters.map(p=>p.name+': '+p.type).join(', ')} → ${compiled.program.resultType}.\nTypeScript checker only; formal verdict not run.`;
    if(gpu){runtime??=await GpuRuntime.create();const report=await runBend(runtime,compiled,JSON.parse($('inputs').value),{arenaWords:Number($('arena').value),maxSteps:Number($('steps').value)});table(report);$('status').textContent+=`\n${report.status.filter(x=>x===0).length}/${report.status.length} GPU results completed in ${report.elapsedMs.toFixed(2)} ms (dispatch + readback).`;if(report.status.some(Boolean))$('status').className='error';window.__bendLastRun=report;}
  } catch(e){$('status').className='error';$('status').textContent=e.message;window.__bendLastError=e.message;}
  finally{controls.forEach(c=>c.disabled=false);}
}
function download(name,text){const u=URL.createObjectURL(new Blob([text],{type:'text/plain'}));const a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
$('example').onchange=loadExample;
$('source').oninput=invalidate;$('entry').oninput=invalidate;
$('compile').onclick=()=>execute(false);$('run').onclick=()=>execute(true);
$('cuda-tab').onclick=()=>{view='cuda';$('cuda-tab').setAttribute('aria-pressed','true');$('wgsl-tab').setAttribute('aria-pressed','false');showCode();};
$('wgsl-tab').onclick=()=>{view='wgsl';$('cuda-tab').setAttribute('aria-pressed','false');$('wgsl-tab').setAttribute('aria-pressed','true');showCode();};
$('save-source').onclick=()=>download('program.bend',$('source').value);
$('save-code').onclick=()=>{if(compiled)download('bend-runtime.'+(view==='cuda'?'cu':'wgsl'),view==='cuda'?compiled.cuda:compiled.artifact.wgsl);};
loadExample();
